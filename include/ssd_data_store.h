// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <memory>
#include <string>
#include "abstract_data_store.h"
#include "distance.h"

namespace diskann
{

template <typename data_t> class SsdDataStore : public AbstractDataStore<data_t>
{
  public:
    // backing_file_path: 在 NVMe 上预分配的文件路径，e.g. /data/index/vectors.bin
    SsdDataStore(const location_t capacity, const size_t dim, std::unique_ptr<Distance<data_t>> distance_fn,
                 const std::string &backing_file_path);
    virtual ~SsdDataStore();

    virtual location_t load(const std::string &filename) override;
    virtual size_t save(const std::string &filename, const location_t num_points) override;
    virtual size_t get_aligned_dim() const override;

    virtual void populate_data(const data_t *vectors, const location_t num_pts) override;
    virtual void populate_data(const std::string &filename, const size_t offset) override;
    virtual void extract_data_to_bin(const std::string &filename, const location_t num_pts) override;

    virtual void get_vector(const location_t i, data_t *target) const override;
    virtual void set_vector(const location_t i, const data_t *const vector) override;
    virtual void prefetch_vector(const location_t loc) override;

    virtual void move_vectors(const location_t old_location_start, const location_t new_location_start,
                              const location_t num_points) override;
    virtual void copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points) override;

    virtual void preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const override;

    virtual float get_distance(const data_t *preprocessed_query, const location_t loc) const override;
    virtual float get_distance(const location_t loc1, const location_t loc2) const override;
    virtual void get_distance(const data_t *preprocessed_query, const location_t *locations,
                              const uint32_t location_count, float *distances,
                              AbstractScratch<data_t> *scratch) const override;
    virtual void get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                              std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const override;

    virtual location_t calculate_medoid() const override;
    virtual Distance<data_t> *get_dist_fn() const override;
    virtual size_t get_alignment_factor() const override;

  protected:
    virtual location_t expand(const location_t new_size) override;
    virtual location_t shrink(const location_t new_size) override;

  private:
    // 重新映射到 new_capacity 大小；负责 munmap + ftruncate + mmap
    void remap(const location_t new_capacity);

    data_t *_data = nullptr; // 指向 mmap 区域
    size_t _aligned_dim;
    int _fd = -1; // backing file fd
    std::string _backing_file_path;

    std::unique_ptr<Distance<data_t>> _distance_fn;
    std::shared_ptr<float[]> _pre_computed_norms; // 留空，后续按需实现
};

} // namespace diskann