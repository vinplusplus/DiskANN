// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "nvm_graph_store.h"
#include "utils.h" // diskann::cout / cerr, open_file_to_write
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace diskann
{

// =============================================================================
// 构造 / 析构
// =============================================================================

NvmGraphStore::NvmGraphStore(const size_t total_pts, const size_t reserve_graph_degree, const std::string &nvm_path)
    : AbstractGraphStore(total_pts, reserve_graph_degree), _nvm_path(nvm_path)
{
    _slot_size = (1 + reserve_graph_degree) * sizeof(uint32_t);
    const size_t total_nvm_bytes = total_pts * _slot_size;

    // PMEM_FILE_CREATE：文件不存在则创建，已存在则按新 size 截断/扩展
    _nvm_base = static_cast<char *>(
        pmem_map_file(nvm_path.c_str(), total_nvm_bytes, PMEM_FILE_CREATE, 0666, &_nvm_mapped_len, &_is_pmem));

    if (_nvm_base == nullptr)
        throw std::runtime_error("NvmGraphStore: pmem_map_file failed on " + nvm_path + ": " + std::strerror(errno));

    diskann::cout << "NvmGraphStore: mapped " << total_nvm_bytes / (1 << 20) << " MB at "
                  << static_cast<void *>(_nvm_base) << " (is_pmem=" << _is_pmem << ")" << std::endl;

    // DRAM 镜像初始化
    _graph.resize(total_pts);
    for (size_t i = 0; i < total_pts; i++)
        _graph[i].reserve(reserve_graph_degree);
}

NvmGraphStore::~NvmGraphStore()
{
    if (_nvm_base != nullptr)
    {
        pmem_unmap(_nvm_base, _nvm_mapped_len);
        _nvm_base = nullptr;
    }
}

// =============================================================================
// 私有辅助
// =============================================================================

void NvmGraphStore::persist_node(const location_t i)
{
    void *addr = slot_ptr(i);
    if (_is_pmem)
        pmem_persist(addr, _slot_size);
    else
        pmem_msync(addr, _slot_size);
}

// =============================================================================
// 核心接口实现
// =============================================================================

const std::vector<location_t> &NvmGraphStore::get_neighbours(const location_t i) const
{
    // 走 DRAM 镜像，零 NVM 访问
    return _graph.at(i);
}

void NvmGraphStore::add_neighbour(const location_t i, location_t neighbour_id)
{
    // 1. 更新 DRAM 镜像
    _graph[i].emplace_back(neighbour_id);
    if (_max_observed_degree < _graph[i].size())
        _max_observed_degree = static_cast<uint32_t>(_graph[i].size());

    // 2. 更新 NVM 槽（整槽写回，保证原子一致性）
    //    注意：这里选择整槽重写而非只追加最后一个元素
    //    原因：degree 字段必须同步更新，单独 persist degree 和 data 有 torn-write 风险
    uint32_t *slot = slot_ptr(i);
    uint32_t degree = static_cast<uint32_t>(_graph[i].size());
    slot[0] = degree;
    // 只需要更新新增的那个邻居和 degree，但为了简单整槽重写
    std::memcpy(slot + 1, _graph[i].data(), degree * sizeof(uint32_t));
    persist_node(i);
}

void NvmGraphStore::clear_neighbours(const location_t i)
{
    // 1. DRAM
    _graph[i].clear();

    // 2. NVM：只需将 degree 置 0，其余数据是脏数据但不影响正确性
    uint32_t *slot = slot_ptr(i);
    slot[0] = 0u;
    // 只 persist degree 字段即可（4 bytes），不必 flush 整槽
    if (_is_pmem)
        pmem_persist(slot, sizeof(uint32_t));
    else
        pmem_msync(slot, sizeof(uint32_t));
}
void NvmGraphStore::swap_neighbours(const location_t a, location_t b)
{
    // 1. DRAM
    _graph[a].swap(_graph[b]);

    // 2. NVM：两个槽分别重写
    auto write_node = [&](location_t idx) {
        uint32_t *slot = slot_ptr(idx);
        uint32_t degree = static_cast<uint32_t>(_graph[idx].size());
        slot[0] = degree;
        std::memcpy(slot + 1, _graph[idx].data(), degree * sizeof(uint32_t));
        persist_node(idx);
    };
    write_node(a);
    write_node(b);
}

void NvmGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours)
{
    // 1. DRAM
    _graph[i].assign(neighbours.begin(), neighbours.end());
    if (_max_observed_degree < neighbours.size())
        _max_observed_degree = static_cast<uint32_t>(neighbours.size());

    // 2. NVM
    uint32_t *slot = slot_ptr(i);
    uint32_t degree = static_cast<uint32_t>(neighbours.size());
    // 安全检查：degree 不应超过 reserve_graph_degree
    if (degree > get_reserve_graph_degree())
    {
        diskann::cerr << "NvmGraphStore::set_neighbours: degree " << degree << " exceeds reserve "
                      << get_reserve_graph_degree() << " for node " << i << std::endl;
        degree = static_cast<uint32_t>(get_reserve_graph_degree());
    }
    slot[0] = degree;
    std::memcpy(slot + 1, neighbours.data(), degree * sizeof(uint32_t));
    persist_node(i);
}

// =============================================================================
// 图结构管理
// =============================================================================

size_t NvmGraphStore::resize_graph(const size_t new_size)
{
    // NVM 在构造时已按 total_pts 全量分配，无需重新 mmap
    // 只扩展 DRAM 镜像的逻辑大小
    if (new_size > get_total_points())
    {
        // 超出 NVM 预分配容量，理论上不应发生
        throw std::runtime_error("NvmGraphStore::resize_graph: new_size " + std::to_string(new_size) +
                                 " exceeds NVM pre-allocated capacity " + std::to_string(get_total_points()));
    }
    _graph.resize(new_size);
    set_total_points(new_size);
    return _graph.size();
}

void NvmGraphStore::clear_graph()
{
    // DRAM
    for (auto &adj : _graph)
        adj.clear();

    // NVM：将所有槽的 degree 字段置 0
    // 全量 memset 再一次 persist 比逐节点 persist 更快
    const size_t total = get_total_points();
    for (size_t i = 0; i < total; i++)
    {
        uint32_t *slot = slot_ptr(i);
        slot[0] = 0u;
    }
    if (_is_pmem)
        pmem_persist(_nvm_base, total * _slot_size);
    else
        pmem_msync(_nvm_base, total * _slot_size);
}

uint32_t NvmGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

size_t NvmGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

// =============================================================================
// load / store（文件格式与 InMemGraphStore 完全兼容）
// =============================================================================

std::tuple<uint32_t, uint32_t, size_t> NvmGraphStore::load(const std::string &index_path_prefix,
                                                           const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}

int NvmGraphStore::store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                         const uint32_t start)
{
    return save_graph(index_path_prefix, num_points, num_frozen_points, start);
}

std::tuple<uint32_t, uint32_t, size_t> NvmGraphStore::load_impl(const std::string &filename, size_t expected_num_points)
{
    // 文件格式与 InMemGraphStore::load_impl 完全相同（变长格式）
    // 读入后：同步写 DRAM 镜像 + NVM 槽
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.read(reinterpret_cast<char *>(&expected_file_size), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&_max_observed_degree), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&start), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&file_frozen_pts), sizeof(size_t));

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (get_total_points() < expected_num_points)
        this->resize_graph(expected_num_points);

    size_t bytes_read = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
    size_t cc = 0;
    uint32_t nodes_read = 0;

    while (bytes_read != expected_file_size)
    {
        uint32_t k;
        in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
        cc += k;

        std::vector<uint32_t> tmp(k);
        in.read(reinterpret_cast<char *>(tmp.data()), k * sizeof(uint32_t));

        // 同步到 DRAM 镜像
        _graph[nodes_read].swap(tmp);

        // 同步到 NVM 槽（load 阶段批量写，最后统一 persist）
        uint32_t *slot = slot_ptr(nodes_read);
        uint32_t deg = std::min(k, static_cast<uint32_t>(get_reserve_graph_degree()));
        slot[0] = deg;
        std::memcpy(slot + 1, _graph[nodes_read].data(), deg * sizeof(uint32_t));

        if (k > _max_range_of_graph)
            _max_range_of_graph = k;

        bytes_read += sizeof(uint32_t) * (static_cast<size_t>(k) + 1);
        ++nodes_read;
        if (nodes_read % 1000000 == 0)
            diskann::cout << "." << std::flush;
    }

    // 批量 persist（比逐节点 persist 更高效，load 路径可以接受）
    if (_is_pmem)
        pmem_persist(_nvm_base, static_cast<size_t>(nodes_read) * _slot_size);
    else
        pmem_msync(_nvm_base, static_cast<size_t>(nodes_read) * _slot_size);

    diskann::cout << "\ndone. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to "
                  << start << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}

int NvmGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                              const size_t num_frozen_points, const uint32_t start)
{
    // 与 InMemGraphStore::save_graph 逻辑相同，从 DRAM 镜像写文件
    // 文件格式不变，保证与 search_disk_index 等工具兼容
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write(reinterpret_cast<char *>(&index_size), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(&_max_observed_degree), sizeof(uint32_t));
    uint32_t ep_u32 = start;
    out.write(reinterpret_cast<char *>(&ep_u32), sizeof(uint32_t));
    size_t frozen_pts = num_frozen_points;
    out.write(reinterpret_cast<char *>(&frozen_pts), sizeof(size_t));

    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = static_cast<uint32_t>(_graph[i].size());
        out.write(reinterpret_cast<char *>(&GK), sizeof(uint32_t));
        out.write(reinterpret_cast<char *>(_graph[i].data()), GK * sizeof(uint32_t));
        if (_graph[i].size() > max_degree)
            max_degree = GK;
        index_size += static_cast<size_t>(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(0, out.beg);
    out.write(reinterpret_cast<char *>(&index_size), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(&max_degree), sizeof(uint32_t));
    out.close();
    return static_cast<int>(index_size);
}

} // namespace diskann