#include "nvm_vector_wal.h"
#include "utils.h"
#include <cerrno>
#include <stdexcept>
#include <iostream>

namespace diskann
{

NvmVectorWAL::NvmVectorWAL(const std::string &nvm_path, uint32_t dim, uint32_t element_size, uint32_t capacity_entries,
                           bool create_new)
    : _nvm_path(nvm_path), _dim(dim), _element_size(element_size), _capacity(capacity_entries)
{
    _vector_bytes = dim * element_size;
    _entry_size = compute_entry_size(dim, element_size);

    const size_t total_bytes = WAL_HEADER_SIZE + static_cast<size_t>(_entry_size) * capacity_entries;

    _pmem_base = static_cast<char *>(
        pmem_map_file(nvm_path.c_str(), total_bytes, PMEM_FILE_CREATE, 0666, &_mapped_len, &_is_pmem));

    if (_pmem_base == nullptr)
        throw std::runtime_error("NvmVectorWAL: pmem_map_file failed on " + nvm_path + ": " + std::strerror(errno));

    _header = reinterpret_cast<WALHeader *>(_pmem_base);

    if (create_new)
    {
        // 初始化 header
        std::memset(_header, 0, WAL_HEADER_SIZE);
        _header->magic = WAL_MAGIC;
        _header->version = WAL_VERSION;
        _header->dim = dim;
        _header->element_size = element_size;
        _header->entry_size = _entry_size;
        _header->capacity = capacity_entries;
        _header->head = 0;
        _header->tail = 0;
        persist_range(_header, WAL_HEADER_SIZE);
        _tail_shadow = 0;
    }
    else
    {
        // 恢复模式：校验 header，重建 tail
        if (_header->magic != WAL_MAGIC)
            throw std::runtime_error("NvmVectorWAL: invalid magic, file may not be a WAL: " + nvm_path);
        if (_header->dim != dim || _header->element_size != element_size)
            throw std::runtime_error("NvmVectorWAL: dim/element_size mismatch in existing WAL");
        if (_header->capacity != capacity_entries)
            throw std::runtime_error("NvmVectorWAL: capacity mismatch in existing WAL");
        recover_tail();
    }

    diskann::cout << "NvmVectorWAL: mapped " << total_bytes / (1 << 20) << " MB at " << static_cast<void *>(_pmem_base)
                  << " (is_pmem=" << _is_pmem << ", capacity=" << capacity_entries << ", entry_size=" << _entry_size
                  << "B)" << std::endl;
}

NvmVectorWAL::NvmVectorWAL(const std::string &nvm_path) : _nvm_path(nvm_path)
{
    // len=0 + 无 PMEM_FILE_CREATE → 映射整个已有文件
    _pmem_base = static_cast<char *>(pmem_map_file(nvm_path.c_str(), 0, 0, 0666, &_mapped_len, &_is_pmem));
    if (_pmem_base == nullptr)
        throw std::runtime_error("NvmVectorWAL recovery: pmem_map_file failed on " + nvm_path + ": " +
                                 std::strerror(errno));
    _header = reinterpret_cast<WALHeader *>(_pmem_base);
    if (_header->magic != WAL_MAGIC)
        throw std::runtime_error("NvmVectorWAL: invalid magic in " + nvm_path);
    _dim = _header->dim;
    _element_size = _header->element_size;
    _vector_bytes = _dim * _element_size;
    _entry_size = _header->entry_size;
    _capacity = _header->capacity;
    recover_tail();
    diskann::cout << "NvmVectorWAL [recovery]: dim=" << _dim << " capacity=" << _capacity << " head=" << _header->head
                  << " tail=" << _tail_shadow << " valid=" << (_tail_shadow - _header->head) << std::endl;
}

NvmVectorWAL::~NvmVectorWAL()
{
    if (_pmem_base != nullptr)
    {
        // 持久化最终 tail
        _header->tail = _tail_shadow;
        persist_range(&_header->tail, sizeof(uint64_t));
        pmem_unmap(_pmem_base, _mapped_len);
        _pmem_base = nullptr;
        _header = nullptr;
    }
}

void NvmVectorWAL::persist_range(void *addr, size_t len)
{
    if (_is_pmem)
        pmem_persist(addr, len);
    else
        pmem_msync(addr, len);
}

bool NvmVectorWAL::append(uint32_t slot_id, const void *vector_data)
{
    std::lock_guard<std::mutex> lock(_append_mutex);

    if (_tail_shadow - _header->head >= _capacity)
        return false; // WAL 满

    const uint64_t seq = _tail_shadow;
    const uint32_t idx = static_cast<uint32_t>(seq % _capacity);
    char *entry = entry_ptr(idx);

    // 写 entry header
    *reinterpret_cast<uint64_t *>(entry) = seq;
    *reinterpret_cast<uint32_t *>(entry + 8) = slot_id;
    *reinterpret_cast<uint32_t *>(entry + 12) = 0; // reserved

    // 写向量数据
    std::memcpy(entry + WAL_ENTRY_HEADER_SIZE, vector_data, _vector_bytes);

    // persist 整个 entry
    persist_range(entry, _entry_size);

    _tail_shadow++;
    // 注意：不 persist header->tail，由 checkpoint 统一做
    return true;
}

void NvmVectorWAL::checkpoint()
{
    std::lock_guard<std::mutex> lock(_append_mutex);
    if (_header->head == _tail_shadow)
        return; // 已经是最新，无需重复 persist
    _header->head = _tail_shadow;
    _header->tail = _tail_shadow;
    persist_range(_header, WAL_HEADER_SIZE);
}

bool NvmVectorWAL::needs_checkpoint() const
{
    // 不加锁，容忍轻微不一致（用于提示性检查）
    uint64_t used = _tail_shadow - _header->head;
    return used >= static_cast<uint64_t>(_capacity) * 9 / 10;
}

uint64_t NvmVectorWAL::num_valid_entries() const
{
    return _tail_shadow - _header->head;
}

void NvmVectorWAL::iterate(const std::function<void(uint32_t slot_id, const void *vector_data)> &callback) const
{
    for (uint64_t seq = _header->head; seq < _tail_shadow; seq++)
    {
        const uint32_t idx = static_cast<uint32_t>(seq % _capacity);
        const char *entry = entry_ptr(idx);

        const uint64_t stored_seq = *reinterpret_cast<const uint64_t *>(entry);
        if (stored_seq != seq)
        {
            // 序列号不匹配，说明此 entry 未成功写入，截断
            diskann::cerr << "NvmVectorWAL::iterate: seq mismatch at expected=" << seq << " found=" << stored_seq
                          << ", truncating" << std::endl;
            break;
        }

        const uint32_t slot_id = *reinterpret_cast<const uint32_t *>(entry + 8);
        const void *vec = entry + WAL_ENTRY_HEADER_SIZE;
        callback(slot_id, vec);
    }
}

void NvmVectorWAL::recover_tail()
{
    // 从 header->head 开始顺序扫描，找到最后一个有效 entry
    uint64_t seq = _header->head;
    while (seq - _header->head < _capacity)
    {
        const uint32_t idx = static_cast<uint32_t>(seq % _capacity);
        const char *entry = entry_ptr(idx);
        const uint64_t stored_seq = *reinterpret_cast<const uint64_t *>(entry);
        if (stored_seq != seq)
            break;
        seq++;
    }
    _tail_shadow = seq;
    // 将真实 tail 写回 NVM header
    _header->tail = seq;
    persist_range(&_header->tail, sizeof(uint64_t));

    diskann::cout << "NvmVectorWAL: recovered tail=" << seq << " (head=" << _header->head
                  << ", valid_entries=" << (seq - _header->head) << ")" << std::endl;
}

} // namespace diskann