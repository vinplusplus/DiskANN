#include "nvm_slot_meta_store.h"
#include "utils.h"
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace diskann
{

NvmSlotMetaStore::NvmSlotMetaStore(size_t total_slots, const std::string &nvm_path, bool create_new)
    : _total_slots(total_slots), _nvm_path(nvm_path)
{
    const size_t total_bytes = total_slots * sizeof(SlotMeta);

    _pmem_base = static_cast<char *>(
        pmem_map_file(nvm_path.c_str(), total_bytes, PMEM_FILE_CREATE, 0666, &_mapped_len, &_is_pmem));

    if (_pmem_base == nullptr)
        throw std::runtime_error("NvmSlotMetaStore: pmem_map_file failed on " + nvm_path + ": " + std::strerror(errno));

    _meta = reinterpret_cast<SlotMeta *>(_pmem_base);

    if (create_new)
    {
        // 全部初始化为 0（EMPTY=0, tag=0）
        std::memset(_pmem_base, 0, total_bytes);
        if (_is_pmem)
            pmem_persist(_pmem_base, total_bytes);
        else
            pmem_msync(_pmem_base, total_bytes);
    }

    diskann::cout << "NvmSlotMetaStore: mapped " << total_bytes / 1024 << " KB at " << static_cast<void *>(_pmem_base)
                  << " (is_pmem=" << _is_pmem << ", slots=" << total_slots << ")" << std::endl;
}

NvmSlotMetaStore::~NvmSlotMetaStore()
{
    if (_pmem_base != nullptr)
    {
        pmem_unmap(_pmem_base, _mapped_len);
        _pmem_base = nullptr;
        _meta = nullptr;
    }
}

void NvmSlotMetaStore::persist_slot(uint32_t slot)
{
    void *addr = slot_ptr(slot);
    if (_is_pmem)
        pmem_persist(addr, sizeof(SlotMeta));
    else
        pmem_msync(addr, sizeof(SlotMeta));
}

void NvmSlotMetaStore::set_status(uint32_t slot, SlotStatus status)
{
    SlotMeta *m = slot_ptr(slot);
    m->status = static_cast<uint16_t>(status);
    if (status == SlotStatus::EMPTY)
        m->tag = UINT32_MAX;
    persist_slot(slot);
}

void NvmSlotMetaStore::set_tag_and_status(uint32_t slot, uint32_t tag, SlotStatus status)
{
    SlotMeta *m = slot_ptr(slot);
    m->tag = tag;
    m->status = static_cast<uint16_t>(status);
    persist_slot(slot);
}

SlotMeta NvmSlotMetaStore::get(uint32_t slot) const
{
    return *slot_ptr(slot);
}

SlotStatus NvmSlotMetaStore::get_status(uint32_t slot) const
{
    return static_cast<SlotStatus>(slot_ptr(slot)->status);
}

uint32_t NvmSlotMetaStore::get_tag(uint32_t slot) const
{
    return slot_ptr(slot)->tag;
}

void NvmSlotMetaStore::scan(const std::function<void(uint32_t, const SlotMeta &)> &callback) const
{
    for (uint32_t i = 0; i < _total_slots; i++)
    {
        const SlotMeta &m = *slot_ptr(i);
        if (static_cast<SlotStatus>(m.status) != SlotStatus::EMPTY)
        {
            callback(i, m);
        }
    }
}

void NvmSlotMetaStore::clear_all()
{
    const size_t total_bytes = _total_slots * sizeof(SlotMeta);
    std::memset(_pmem_base, 0, total_bytes);
    if (_is_pmem)
        pmem_persist(_pmem_base, total_bytes);
    else
        pmem_msync(_pmem_base, total_bytes);
}

} // namespace diskann