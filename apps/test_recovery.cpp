// test_recovery.cpp — Crash recovery for NVM-SSD tiered ANNS index
//
// Reads NVM files (graph, slot metadata, WAL) and SSD vector backing file,
// performs six-phase recovery, writes clean DiskANN index loadable by
// search_memory_index --dynamic true --tags 1.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
#include <libpmem.h>

#include "nvm_slot_meta_store.h"
#include "nvm_vector_wal.h"
#include "timer.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string nvm_graph_path, meta_path, wal_path, ssd_path, output_prefix;
    uint32_t dim, max_points, num_frozen_pts;

    po::options_description desc("Crash Recovery Options");
    desc.add_options()("help", "Print help")("nvm_graph_path", po::value(&nvm_graph_path)->required(),
                                             "NVM graph file")("meta_path", po::value(&meta_path)->required(),
                                                               "NVM slot metadata file")(
        "wal_path", po::value(&wal_path)->required(),
        "NVM vector WAL file")("ssd_path", po::value(&ssd_path)->required(), "SSD vector backing file")(
        "output_prefix", po::value(&output_prefix)->required(), "Output index prefix")(
        "dim", po::value(&dim)->required(), "Vector dimension")("max_points", po::value(&max_points)->required(),
                                                                "max_points used at build time")(
        "num_frozen_pts", po::value(&num_frozen_pts)->default_value(1), "Number of frozen points");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n" << desc << std::endl;
        return 1;
    }

    const uint32_t total_slots = max_points + num_frozen_pts;
    const uint32_t aligned_dim = ((dim + 7) / 8) * 8;

    std::cout << "=== Recovery Parameters ===" << std::endl;
    std::cout << "dim=" << dim << " aligned_dim=" << aligned_dim << " max_points=" << max_points
              << " frozen=" << num_frozen_pts << " total_slots=" << total_slots << std::endl;

    diskann::Timer total_timer;

    // ================================================================
    // Phase 1: Scan SlotMeta — classify every slot
    // ================================================================
    std::cout << "\n=== Phase 1: Scan SlotMeta ===" << std::endl;
    diskann::NvmSlotMetaStore meta(total_slots, meta_path, false /* recovery mode */);

    std::vector<uint32_t> live_regular, frozen_slots;
    std::unordered_set<uint32_t> remove_set;
    std::unordered_map<uint32_t, uint32_t> slot_tag;
    uint32_t n_empty = 0, n_inserting = 0, n_dpend = 0, n_dfin = 0;

    for (uint32_t i = 0; i < total_slots; i++)
    {
        auto m = meta.get(i);
        auto st = static_cast<diskann::SlotStatus>(m.status);
        if (st == diskann::SlotStatus::LIVE)
        {
            slot_tag[i] = m.tag;
            if (i >= max_points)
                frozen_slots.push_back(i);
            else
                live_regular.push_back(i);
        }
        else
        {
            remove_set.insert(i);
            switch (st)
            {
            case diskann::SlotStatus::EMPTY:
                n_empty++;
                break;
            case diskann::SlotStatus::INSERTING:
                n_inserting++;
                break;
            case diskann::SlotStatus::DELETED_PENDING:
                n_dpend++;
                break;
            case diskann::SlotStatus::DELETED_FINISHED:
                n_dfin++;
                break;
            default:
                std::cerr << "WARNING: unknown status " << m.status << " at slot " << i << std::endl;
                break;
            }
        }
    }

    std::cout << "  LIVE(regular)=" << live_regular.size() << "  FROZEN=" << frozen_slots.size()
              << "  EMPTY=" << n_empty << "  INSERTING=" << n_inserting << "  DELETED_PENDING=" << n_dpend
              << "  DELETED_FINISHED=" << n_dfin << std::endl;

    if (n_inserting > 0)
        std::cout << "  -> " << n_inserting << " in-flight inserts will be ROLLED BACK" << std::endl;
    if (n_dpend > 0)
        std::cout << "  -> " << n_dpend << " in-flight deletes will be treated as DELETED_FINISHED" << std::endl;

    // ================================================================
    // Phase 2: Replay WAL → recover lost SSD vectors
    // ================================================================
    std::cout << "\n=== Phase 2: Replay WAL ===" << std::endl;

    // Open SSD file (read-write, NO truncate)
    int ssd_fd = open(ssd_path.c_str(), O_RDWR);
    if (ssd_fd < 0)
    {
        std::cerr << "Cannot open SSD file: " << ssd_path << ": " << strerror(errno) << std::endl;
        return 1;
    }
    struct stat ssd_stat;
    fstat(ssd_fd, &ssd_stat);
    size_t ssd_size = ssd_stat.st_size;
    float *ssd_data = static_cast<float *>(mmap(nullptr, ssd_size, PROT_READ | PROT_WRITE, MAP_SHARED, ssd_fd, 0));
    if (ssd_data == MAP_FAILED)
    {
        std::cerr << "Cannot mmap SSD file: " << strerror(errno) << std::endl;
        return 1;
    }

    std::unordered_set<uint32_t> live_set;
    for (auto s : live_regular)
        live_set.insert(s);
    for (auto s : frozen_slots)
        live_set.insert(s);

    uint32_t wal_replayed = 0, wal_skipped = 0;
    diskann::NvmVectorWAL wal(wal_path); // recovery constructor

    wal.iterate([&](uint32_t slot_id, const void *vec) {
        if (live_set.count(slot_id))
        {
            float *dst = ssd_data + static_cast<size_t>(slot_id) * aligned_dim;
            std::memcpy(dst, vec, dim * sizeof(float));
            if (aligned_dim > dim)
                std::memset(dst + dim, 0, (aligned_dim - dim) * sizeof(float));
            wal_replayed++;
        }
        else
        {
            wal_skipped++; // rolled-back insert or already-deleted
        }
    });

    msync(ssd_data, ssd_size, MS_SYNC);
    std::cout << "  WAL entries: replayed=" << wal_replayed << " skipped=" << wal_skipped << "  SSD synced."
              << std::endl;

    // ================================================================
    // Phase 3+4+5: Read NVM Graph, clean dangling edges, compact
    // ================================================================
    std::cout << "\n=== Phase 3-5: Read graph, clean edges, compact ===" << std::endl;

    // Open NVM graph (read-only, no PMEM_FILE_CREATE)
    size_t graph_mapped_len;
    int graph_is_pmem;
    char *graph_base =
        static_cast<char *>(pmem_map_file(nvm_graph_path.c_str(), 0, 0, 0, &graph_mapped_len, &graph_is_pmem));
    if (!graph_base)
    {
        std::cerr << "Cannot open NVM graph: " << nvm_graph_path << ": " << strerror(errno) << std::endl;
        return 1;
    }

    // Infer slot_size and reserve_degree from file size
    uint32_t slot_size = static_cast<uint32_t>(graph_mapped_len / total_slots);
    uint32_t reserve_degree = slot_size / sizeof(uint32_t) - 1;
    std::cout << "  NVM graph: " << graph_mapped_len / (1 << 20) << " MB, slot_size=" << slot_size
              << "B, reserve_degree=" << reserve_degree << std::endl;

    if (static_cast<size_t>(slot_size) * total_slots != graph_mapped_len)
    {
        std::cerr << "WARNING: graph file size not exactly divisible by total_slots" << std::endl;
    }

    // Build compacted layout: [live_regular sorted | frozen_slots]
    std::sort(live_regular.begin(), live_regular.end());
    uint32_t num_live = static_cast<uint32_t>(live_regular.size());
    uint32_t num_frozen = static_cast<uint32_t>(frozen_slots.size());
    uint32_t new_total = num_live + num_frozen;
    uint32_t new_start = num_live; // frozen point position in new layout

    // old slot → new position
    std::unordered_map<uint32_t, uint32_t> slot_to_new;
    slot_to_new.reserve(new_total);
    for (uint32_t i = 0; i < num_live; i++)
        slot_to_new[live_regular[i]] = i;
    for (uint32_t i = 0; i < num_frozen; i++)
        slot_to_new[frozen_slots[i]] = num_live + i;

    // Build compacted adjacency lists, remapping IDs and removing dangling edges
    struct CompactAdj
    {
        std::vector<uint32_t> nbrs;
    };
    std::vector<CompactAdj> new_graph(new_total);
    uint32_t max_degree = 0;
    uint64_t dangling_removed = 0;
    uint64_t total_edges = 0;

    auto process_slot = [&](uint32_t old_slot, uint32_t new_pos) {
        const uint32_t *p = reinterpret_cast<const uint32_t *>(graph_base + static_cast<size_t>(old_slot) * slot_size);
        uint32_t deg = p[0];
        auto &adj = new_graph[new_pos].nbrs;
        adj.reserve(deg);
        for (uint32_t j = 1; j <= deg; j++)
        {
            uint32_t nbr = p[j];
            if (remove_set.count(nbr))
            {
                dangling_removed++;
                continue;
            }
            auto it = slot_to_new.find(nbr);
            if (it != slot_to_new.end())
            {
                adj.push_back(it->second);
            }
            else
            {
                dangling_removed++;
            }
        }
        total_edges += adj.size();
        if (adj.size() > max_degree)
            max_degree = static_cast<uint32_t>(adj.size());
    };

    for (uint32_t i = 0; i < num_live; i++)
        process_slot(live_regular[i], i);
    for (uint32_t i = 0; i < num_frozen; i++)
        process_slot(frozen_slots[i], num_live + i);

    // Count zero-degree nodes (excluding frozen)
    uint32_t zero_degree = 0;
    for (uint32_t i = 0; i < num_live; i++)
        if (new_graph[i].nbrs.empty())
            zero_degree++;

    std::cout << "  Compacted: " << new_total << " nodes (" << num_live << " regular + " << num_frozen << " frozen)"
              << std::endl;
    std::cout << "  Total edges: " << total_edges << "  Dangling removed: " << dangling_removed
              << "  Max degree: " << max_degree << "  Zero-degree: " << zero_degree << std::endl;

    // ================================================================
    // Phase 6: Write output files
    // ================================================================
    std::cout << "\n=== Phase 6: Write output files ===" << std::endl;

    // --- Graph file: {output_prefix} ---
    {
        std::string graph_file = output_prefix;
        std::ofstream out(graph_file, std::ios::binary);
        if (!out)
        {
            std::cerr << "Cannot create " << graph_file << std::endl;
            return 1;
        }

        // Calculate file size (header + all adjacency lists)
        size_t file_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
        for (auto &adj : new_graph)
            file_size += sizeof(uint32_t) + adj.nbrs.size() * sizeof(uint32_t);

        size_t nfp = num_frozen;
        out.write(reinterpret_cast<char *>(&file_size), sizeof(size_t));
        out.write(reinterpret_cast<char *>(&max_degree), sizeof(uint32_t));
        out.write(reinterpret_cast<char *>(&new_start), sizeof(uint32_t));
        out.write(reinterpret_cast<char *>(&nfp), sizeof(size_t));

        for (auto &adj : new_graph)
        {
            uint32_t deg = static_cast<uint32_t>(adj.nbrs.size());
            out.write(reinterpret_cast<char *>(&deg), sizeof(uint32_t));
            if (deg > 0)
                out.write(reinterpret_cast<const char *>(adj.nbrs.data()), deg * sizeof(uint32_t));
        }
        out.close();
        std::cout << "  Graph: " << graph_file << " (" << file_size << " bytes)" << std::endl;
    }

    // --- Data file: {output_prefix}.data ---
    {
        std::string data_file = output_prefix + ".data";
        std::ofstream out(data_file, std::ios::binary);
        if (!out)
        {
            std::cerr << "Cannot create " << data_file << std::endl;
            return 1;
        }
        uint32_t npts_out = new_total;
        uint32_t ndim_out = dim;
        out.write(reinterpret_cast<char *>(&npts_out), sizeof(uint32_t));
        out.write(reinterpret_cast<char *>(&ndim_out), sizeof(uint32_t));

        // Write vectors in compacted order (original dim, no alignment padding)
        for (uint32_t i = 0; i < num_live; i++)
        {
            const float *src = ssd_data + static_cast<size_t>(live_regular[i]) * aligned_dim;
            out.write(reinterpret_cast<const char *>(src), dim * sizeof(float));
        }
        for (uint32_t i = 0; i < num_frozen; i++)
        {
            const float *src = ssd_data + static_cast<size_t>(frozen_slots[i]) * aligned_dim;
            out.write(reinterpret_cast<const char *>(src), dim * sizeof(float));
        }
        out.close();
        std::cout << "  Data: " << data_file << " (" << npts_out << " points × " << ndim_out << "d)" << std::endl;
    }

    // --- Tags file: {output_prefix}.tags ---
    {
        std::string tags_file = output_prefix + ".tags";
        std::ofstream out(tags_file, std::ios::binary);
        if (!out)
        {
            std::cerr << "Cannot create " << tags_file << std::endl;
            return 1;
        }
        uint32_t npts_out = new_total;
        uint32_t one = 1;
        out.write(reinterpret_cast<char *>(&npts_out), sizeof(uint32_t));
        out.write(reinterpret_cast<char *>(&one), sizeof(uint32_t));

        for (uint32_t i = 0; i < num_live; i++)
        {
            uint32_t tag = slot_tag[live_regular[i]];
            out.write(reinterpret_cast<char *>(&tag), sizeof(uint32_t));
        }
        for (uint32_t i = 0; i < num_frozen; i++)
        {
            uint32_t tag = 0; // frozen point tag = 0
            out.write(reinterpret_cast<char *>(&tag), sizeof(uint32_t));
        }
        out.close();
        std::cout << "  Tags: " << tags_file << " (" << npts_out << " entries)" << std::endl;
    }

    // Cleanup
    pmem_unmap(graph_base, graph_mapped_len);
    munmap(ssd_data, ssd_size);
    close(ssd_fd);

    double total_sec = total_timer.elapsed() / 1000000.0;
    std::cout << "\n=== Recovery complete in " << total_sec << " seconds ===" << std::endl;
    std::cout << "  Active points: " << num_live << " + " << num_frozen << " frozen" << std::endl;
    std::cout << "  Rolled back:   " << n_inserting << " in-flight inserts" << std::endl;
    std::cout << "  Cleaned up:    " << (n_dpend + n_dfin) << " pending/finished deletes" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "  1. Compute GT:  ./apps/utils/compute_groundtruth --data_type float --dist_fn l2 \\" << std::endl;
    std::cout << "       --base_file " << output_prefix << ".data \\" << std::endl;
    std::cout << "       --query_file <query.fbin> --gt_file <gt_file> -K 100" << std::endl;
    std::cout << "  2. Convert GT position→tag (see 坑12)" << std::endl;
    std::cout << "  3. Search:      ./apps/search_memory_index --data_type float --dist_fn l2 \\" << std::endl;
    std::cout << "       --index_path_prefix " << output_prefix << " \\" << std::endl;
    std::cout << "       --dynamic true --tags 1 ..." << std::endl;

    return 0;
}