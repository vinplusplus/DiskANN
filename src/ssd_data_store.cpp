// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ssd_data_store.h"
#include "abstract_scratch.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace diskann
{

// ─────────────────────────────────────────────
// 内部工具
// ─────────────────────────────────────────────
namespace
{

// 将 fd 对应文件截断/扩展到 byte_size，然后 mmap MAP_SHARED 返回指针
// 失败直接 throw
void *mmap_file(int fd, size_t byte_size)
{
    if (ftruncate(fd, static_cast<off_t>(byte_size)) != 0)
    {
        throw diskann::ANNException("SsdDataStore: ftruncate failed: " + std::string(strerror(errno)), -1);
    }
    void *ptr = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED)
    {
        throw diskann::ANNException("SsdDataStore: mmap failed: " + std::string(strerror(errno)), -1);
    }
    // 提示内核：访问模式为随机，不预读
    madvise(ptr, byte_size, MADV_RANDOM);
    return ptr;
}

} // anonymous namespace

// ─────────────────────────────────────────────
// 构造 / 析构
// ─────────────────────────────────────────────
template <typename data_t>
SsdDataStore<data_t>::SsdDataStore(const location_t capacity, const size_t dim,
                                   std::unique_ptr<Distance<data_t>> distance_fn, const std::string &backing_file_path)
    : AbstractDataStore<data_t>(capacity, dim), _backing_file_path(backing_file_path),
      _distance_fn(std::move(distance_fn))
{
    _aligned_dim = ROUND_UP(dim, _distance_fn->get_required_alignment());

    // 创建或覆盖 backing file
    _fd = open(backing_file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (_fd < 0)
    {
        throw diskann::ANNException(
            "SsdDataStore: cannot open backing file " + backing_file_path + ": " + strerror(errno), -1);
    }

    size_t byte_size = static_cast<size_t>(capacity) * _aligned_dim * sizeof(data_t);
    _data = reinterpret_cast<data_t *>(mmap_file(_fd, byte_size));
    // mmap 后内核保证新页清零（O_TRUNC 截断），无需 memset
}

template <typename data_t> SsdDataStore<data_t>::~SsdDataStore()
{
    if (_data != nullptr)
    {
        size_t byte_size = static_cast<size_t>(this->_capacity) * _aligned_dim * sizeof(data_t);
        munmap(_data, byte_size);
        _data = nullptr;
    }
    if (_fd >= 0)
    {
        close(_fd);
        _fd = -1;
    }
}

// ─────────────────────────────────────────────
// 内部：重新映射到新容量
// ─────────────────────────────────────────────
template <typename data_t> void SsdDataStore<data_t>::remap(const location_t new_capacity)
{
    size_t old_byte_size = static_cast<size_t>(this->_capacity) * _aligned_dim * sizeof(data_t);
    size_t new_byte_size = static_cast<size_t>(new_capacity) * _aligned_dim * sizeof(data_t);

    // msync 保证已写数据落盘，再 munmap
    msync(_data, old_byte_size, MS_SYNC);
    munmap(_data, old_byte_size);
    _data = nullptr;

    _data = reinterpret_cast<data_t *>(mmap_file(_fd, new_byte_size));
    this->_capacity = new_capacity;
}

// ─────────────────────────────────────────────
// load / save
// ─────────────────────────────────────────────
template <typename data_t> location_t SsdDataStore<data_t>::load(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        throw diskann::ANNException("SsdDataStore::load: file not found: " + filename, -1);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        throw diskann::ANNException("SsdDataStore::load: dim mismatch: file=" + std::to_string(file_dim) +
                                        " store=" + std::to_string(this->_dim),
                                    -1);
    }
    if (file_num_points > this->capacity())
    {
        this->resize(static_cast<location_t>(file_num_points));
    }

    // 复用 InMemDataStore 的对齐拷贝工具函数
    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);
    return static_cast<location_t>(file_num_points);
}

template <typename data_t> size_t SsdDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    // 同 InMemDataStore：以原始维度写出（去掉对齐 padding）
    return save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

// ─────────────────────────────────────────────
// 维度查询
// ─────────────────────────────────────────────
template <typename data_t> size_t SsdDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

template <typename data_t> size_t SsdDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

// ─────────────────────────────────────────────
// populate / extract
// ─────────────────────────────────────────────
template <typename data_t> void SsdDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    memset(_data, 0, _aligned_dim * sizeof(data_t) * num_pts);
    for (location_t i = 0; i < num_pts; i++)
    {
        std::memmove(_data + i * _aligned_dim, vectors + i * this->_dim, this->_dim * sizeof(data_t));
    }
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, _aligned_dim, num_pts);
    }
}

template <typename data_t> void SsdDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    size_t npts, ndim;
    copy_aligned_data_from_file(filename.c_str(), _data, npts, ndim, _aligned_dim, offset);

    if (static_cast<location_t>(npts) > this->capacity())
    {
        throw diskann::ANNException("SsdDataStore::populate_data: npts > capacity", -1);
    }
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, _aligned_dim, this->capacity());
    }
}

template <typename data_t>
void SsdDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_points)
{
    save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

// ─────────────────────────────────────────────
// 单点读写（流式更新热路径）
// ─────────────────────────────────────────────
template <typename data_t> void SsdDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    memcpy(dest, _data + i * _aligned_dim, this->_dim * sizeof(data_t));
}

template <typename data_t> void SsdDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    size_t offset = loc * _aligned_dim;
    memset(_data + offset, 0, _aligned_dim * sizeof(data_t));
    memcpy(_data + offset, vector, this->_dim * sizeof(data_t));
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data + offset, _aligned_dim, 1);
    }
    // 为了和NVM上的图拓扑维持一致性，需要类似NVM的即时持久化
    // 持久化：与 NvmGraphStore 的 pmem_persist 对称
    // size_t byte_offset = offset * sizeof(data_t);
    // size_t page_start = byte_offset & ~(size_t)4095; // 4KB 对齐
    // size_t page_end = (byte_offset + _aligned_dim * sizeof(data_t) + 4095) & ~(size_t)4095;
    // msync(reinterpret_cast<char *>(_data) + page_start, page_end - page_start, MS_SYNC);
}

template <typename data_t> void SsdDataStore<data_t>::flush()
{
    size_t byte_size = static_cast<size_t>(this->_capacity) * _aligned_dim * sizeof(data_t);
    if (_data != nullptr && byte_size > 0)
    {
        if (msync(_data, byte_size, MS_SYNC) != 0)
        {
            diskann::cerr << "SsdDataStore::flush: msync failed: " << strerror(errno) << std::endl;
        }
    }
}

template <typename data_t> void SsdDataStore<data_t>::prefetch_vector(const location_t loc)
{
    diskann::prefetch_vector(reinterpret_cast<const char *>(_data) +
                                 _aligned_dim * static_cast<size_t>(loc) * sizeof(data_t),
                             sizeof(data_t) * _aligned_dim);
}

// ─────────────────────────────────────────────
// 向量搬移（consolidate_deletes 调用路径）
// ─────────────────────────────────────────────
template <typename data_t>
void SsdDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points)
{
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    memmove(_data + _aligned_dim * to_loc, _data + _aligned_dim * from_loc, num_points * _aligned_dim * sizeof(data_t));
}

template <typename data_t>
void SsdDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                        const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
        return;

    uint32_t clear_start = old_location_start;
    uint32_t clear_end = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        if (clear_start < new_location_start + num_locations)
            clear_start = new_location_start + num_locations;
    }
    else
    {
        if (clear_end > new_location_start)
            clear_end = new_location_start;
    }

    copy_vectors(old_location_start, new_location_start, num_locations);
    memset(_data + _aligned_dim * clear_start, 0, sizeof(data_t) * _aligned_dim * (clear_end - clear_start));
}

// ─────────────────────────────────────────────
// 查询预处理
// ─────────────────────────────────────────────
template <typename data_t>
void SsdDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
    if (query_scratch == nullptr)
    {
        throw diskann::ANNException("SsdDataStore::preprocess_query: scratch is null", -1);
    }
    memcpy(query_scratch->aligned_query_T(), query, sizeof(data_t) * this->get_dims());
}

// ─────────────────────────────────────────────
// 距离计算（直接用 mmap 指针，无额外拷贝）
// ─────────────────────────────────────────────
template <typename data_t> float SsdDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    return _distance_fn->compare(query, _data + _aligned_dim * loc, static_cast<uint32_t>(_aligned_dim));
}

template <typename data_t> float SsdDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    return _distance_fn->compare(_data + loc1 * _aligned_dim, _data + loc2 * _aligned_dim,
                                 static_cast<uint32_t>(_aligned_dim));
}

template <typename data_t>
void SsdDataStore<data_t>::get_distance(const data_t *query, const location_t *locations, const uint32_t location_count,
                                        float *distances, AbstractScratch<data_t> *) const
{
    for (uint32_t i = 0; i < location_count; i++)
    {
        distances[i] =
            _distance_fn->compare(query, _data + locations[i] * _aligned_dim, static_cast<uint32_t>(_aligned_dim));
    }
}

template <typename data_t>
void SsdDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                        std::vector<float> &distances, AbstractScratch<data_t> *) const
{
    for (size_t i = 0; i < ids.size(); i++)
    {
        distances[i] = _distance_fn->compare(preprocessed_query, _data + ids[i] * _aligned_dim,
                                             static_cast<uint32_t>(_aligned_dim));
    }
}

// ─────────────────────────────────────────────
// medoid（静态构建路径，流式场景不使用）
// ─────────────────────────────────────────────
template <typename data_t> location_t SsdDataStore<data_t>::calculate_medoid() const
{
    std::vector<float> center(_aligned_dim, 0.0f);
    for (size_t i = 0; i < this->capacity(); i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += static_cast<float>(_data[i * _aligned_dim + j]);
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= static_cast<float>(this->capacity());

    std::vector<float> distances(this->capacity());
    for (size_t i = 0; i < this->capacity(); i++)
    {
        float dist = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            float diff = center[j] - static_cast<float>(_data[i * _aligned_dim + j]);
            dist += diff * diff;
        }
        distances[i] = dist;
    }
    return static_cast<location_t>(std::min_element(distances.begin(), distances.end()) - distances.begin());
}

template <typename data_t> Distance<data_t> *SsdDataStore<data_t>::get_dist_fn() const
{
    return _distance_fn.get();
}

// ─────────────────────────────────────────────
// expand / shrink（流式场景 resize 路径）
// ─────────────────────────────────────────────
template <typename data_t> location_t SsdDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size <= this->capacity())
    {
        throw diskann::ANNException("SsdDataStore::expand: new_size <= current capacity", -1);
    }
    remap(new_size);
    return this->_capacity;
}

template <typename data_t> location_t SsdDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size >= this->capacity())
    {
        throw diskann::ANNException("SsdDataStore::shrink: new_size >= current capacity", -1);
    }
    remap(new_size);
    return this->_capacity;
}

// ─────────────────────────────────────────────
// 显式实例化
// ─────────────────────────────────────────────
template class SsdDataStore<float>;
template class SsdDataStore<int8_t>;
template class SsdDataStore<uint8_t>;

} // namespace diskann