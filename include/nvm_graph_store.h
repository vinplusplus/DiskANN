// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "abstract_graph_store.h"
#include <libpmem.h>
#include <stdexcept>
#include <string>

namespace diskann
{

class NvmGraphStore : public AbstractGraphStore
{
  public:
    // nvm_path: 文件路径，例如 "/mnt/pmem0/graph.bin"
    // 构造时按 total_pts × slot_size 一次性 mmap，文件不存在则创建
    NvmGraphStore(const size_t total_pts, const size_t reserve_graph_degree, const std::string &nvm_path);
    ~NvmGraphStore();

    // ---- AbstractGraphStore interface ----
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                        const size_t num_points) override;
    virtual int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                      const uint32_t start) override;

    virtual const std::vector<location_t> &get_neighbours(const location_t i) const override;
    virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
    virtual void clear_neighbours(const location_t i) override;
    virtual void swap_neighbours(const location_t a, location_t b) override;
    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbours) override;

    virtual size_t resize_graph(const size_t new_size) override;
    virtual void clear_graph() override;

    virtual uint32_t get_max_observed_degree() override;
    virtual size_t get_max_range_of_graph() override;

  private:
    // ---- NVM 区域 ----
    char *_nvm_base = nullptr;  // pmem_map_file 返回的基地址
    size_t _nvm_mapped_len = 0; // 实际 mmap 大小（pmem_map_file 写出）
    int _is_pmem = 0;           // 1 = 真 PMem，0 = 普通 mmap（回退）
    std::string _nvm_path;
    size_t _slot_size = 0; // bytes per node = (1 + R) * 4

    // ---- DRAM 镜像（搜索热路径） ----
    std::vector<std::vector<uint32_t>> _graph;

    // ---- 元数据 ----
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;

    // ---- 私有辅助 ----

    // 返回节点 i 在 NVM 上的槽起始地址（uint32_t*）
    inline uint32_t *slot_ptr(const location_t i) const
    {
        return reinterpret_cast<uint32_t *>(_nvm_base + static_cast<size_t>(i) * _slot_size);
    }

    // 将节点 i 的整个槽 flush 到持久域
    // 如果 _is_pmem==1 用 pmem_persist，否则 pmem_msync（在普通 SSD 上退化为 msync）
    void persist_node(const location_t i);

    // load/store 的实际实现（与 InMemGraphStore 接口一致，文件格式兼容）
    std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);
    int save_graph(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                   const uint32_t start);
};

} // namespace diskann