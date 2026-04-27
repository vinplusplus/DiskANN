// nvm_slot_meta_store.h — NVM-persistent per-slot metadata for crash recovery
#pragma once

#include <libpmem.h>
#include <cstdint>
#include <functional>
#include <string>

namespace diskann
{

enum class SlotStatus : uint16_t
{
    EMPTY = 0,            // 空闲，可分配
    INSERTING = 1,        // 插入进行中，崩溃时回滚
    LIVE = 2,             // 正常活跃节点
    DELETED_PENDING = 3,  // 删除修复进行中，崩溃时继续执行
    DELETED_FINISHED = 4, // 单点删除完成，等待全局清悬挂边
};

struct SlotMeta
{
    uint32_t tag;    // 外部标识符，UINT32_MAX = 无效
    uint16_t status; // SlotStatus
    uint16_t reserved;
};
static_assert(sizeof(SlotMeta) == 8, "SlotMeta must be 8 bytes");

class NvmSlotMetaStore
{
  public:
    /// @param total_slots  max_points + num_frozen_pts
    /// @param nvm_path     e.g. "/mnt/pmem0/meta.bin"
    /// @param create_new   true=初始化为全 EMPTY; false=保留 NVM 上已有内容（恢复模式）
    NvmSlotMetaStore(size_t total_slots, const std::string &nvm_path, bool create_new = true);
    ~NvmSlotMetaStore();

    // --- 单字段更新 ---
    void set_status(uint32_t slot, SlotStatus status);

    // --- 组合更新 ---
    void set_tag_and_status(uint32_t slot, uint32_t tag, SlotStatus status);

    // --- 读取 ---
    SlotMeta get(uint32_t slot) const;
    SlotStatus get_status(uint32_t slot) const;
    uint32_t get_tag(uint32_t slot) const;

    // --- 批量操作 ---
    size_t total_slots() const
    {
        return _total_slots;
    }

    /// 遍历所有非 EMPTY 的 slot，调用 callback
    void scan(const std::function<void(uint32_t slot, const SlotMeta &meta)> &callback) const;

    /// 全部重置为 EMPTY（仅用于测试/初始化）
    void clear_all();

  private:
    SlotMeta *_meta = nullptr;
    char *_pmem_base = nullptr;
    size_t _mapped_len = 0;
    int _is_pmem = 0;
    size_t _total_slots = 0;
    std::string _nvm_path;

    inline SlotMeta *slot_ptr(uint32_t slot) const
    {
        return _meta + slot;
    }

    void persist_slot(uint32_t slot);
};

} // namespace diskann