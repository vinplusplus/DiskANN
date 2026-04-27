// nvm_vector_wal.h — NVM-persistent circular WAL for vector data
// 保护 SSD page cache 中尚未落盘的向量，崩溃后可重放恢复
#pragma once

#include <libpmem.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>

namespace diskann
{

// WAL 文件布局:
//   [WALHeader : 64B]
//   [Entry 0   : entry_size B]
//   [Entry 1   : entry_size B]
//   ...
//   [Entry cap-1 : entry_size B]
//
// Entry 布局 (对齐到 64B):
//   [seq : uint64_t]      偏移 0
//   [slot_id : uint32_t]  偏移 8
//   [reserved : uint32_t] 偏移 12
//   [vector_data : dim * element_size bytes]  偏移 16
//   [padding]

static constexpr uint64_t WAL_MAGIC = 0xDEADBEEFCAFE0001ULL;
static constexpr uint32_t WAL_VERSION = 1;
static constexpr size_t WAL_HEADER_SIZE = 64;
static constexpr size_t WAL_ENTRY_HEADER_SIZE = 16; // seq(8) + slot_id(4) + reserved(4)
static constexpr size_t WAL_ALIGN = 64;             // cache line

struct WALHeader
{
    uint64_t magic;        // WAL_MAGIC
    uint32_t version;      // WAL_VERSION
    uint32_t dim;          // 向量维度
    uint32_t element_size; // sizeof(T)
    uint32_t entry_size;   // 对齐后的单条目大小
    uint32_t capacity;     // 环形缓冲区总条目数
    uint32_t reserved;
    uint64_t head; // 最老未 checkpoint 条目的 seq（已 checkpoint 的全部 < head）
    uint64_t tail; // 上次持久化的 tail（恢复用提示，实际 tail 通过扫描确定）
    uint8_t padding[16];
};
static_assert(sizeof(WALHeader) == WAL_HEADER_SIZE, "WALHeader must be 64 bytes");

class NvmVectorWAL
{
  public:
    /// @param nvm_path        e.g. "/mnt/pmem0/wal.bin"
    /// @param dim             向量维度（原始维度, 非 aligned_dim）
    /// @param element_size    sizeof(T)，float=4, uint8=1
    /// @param capacity_entries 环形缓冲区容量（条目数）
    /// @param create_new      true=初始化新 WAL; false=从已有文件恢复
    NvmVectorWAL(const std::string &nvm_path, uint32_t dim, uint32_t element_size, uint32_t capacity_entries,
                 bool create_new = true);
    /// 恢复模式构造函数：从 NVM 文件头自动读取全部参数
    NvmVectorWAL(const std::string &nvm_path);
    ~NvmVectorWAL();

    /// 追加一条 WAL 记录。线程安全（内部 mutex）。
    /// @return true 成功, false WAL 已满（需先 checkpoint）
    bool append(uint32_t slot_id, const void *vector_data);

    /// 推进 head 到当前 tail，释放全部已写条目。
    /// 调用前必须确保 SSD 已 msync（由 caller 保证）。
    void checkpoint();

    /// WAL 使用率 >= 90% 时返回 true
    bool needs_checkpoint() const;

    /// 当前有效条目数
    uint64_t num_valid_entries() const;

    /// 遍历 [head, tail) 的所有有效条目，用于崩溃恢复
    void iterate(const std::function<void(uint32_t slot_id, const void *vector_data)> &callback) const;

    uint32_t entry_size() const
    {
        return _entry_size;
    }
    uint32_t capacity() const
    {
        return _capacity;
    }

  private:
    // NVM 映射
    char *_pmem_base = nullptr;
    size_t _mapped_len = 0;
    int _is_pmem = 0;
    std::string _nvm_path;

    // 逻辑参数
    uint32_t _dim = 0;
    uint32_t _element_size = 0;
    uint32_t _vector_bytes = 0; // dim * element_size
    uint32_t _entry_size = 0;   // 对齐后
    uint32_t _capacity = 0;

    // Header 指针（NVM 上）
    WALHeader *_header = nullptr;

    // DRAM tail 影子（避免每次 append 都 persist header）
    uint64_t _tail_shadow = 0;

    // 并发控制
    mutable std::mutex _append_mutex;

    // 辅助
    inline char *entry_ptr(uint32_t idx) const
    {
        return _pmem_base + WAL_HEADER_SIZE + static_cast<size_t>(idx) * _entry_size;
    }

    void persist_range(void *addr, size_t len);

    /// 恢复模式：从 head 开始扫描，重建 _tail_shadow
    void recover_tail();

    static uint32_t compute_entry_size(uint32_t dim, uint32_t element_size)
    {
        uint32_t raw = WAL_ENTRY_HEADER_SIZE + dim * element_size;
        return (raw + WAL_ALIGN - 1) / WAL_ALIGN * WAL_ALIGN;
    }
};

} // namespace diskann