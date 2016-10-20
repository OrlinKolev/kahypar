/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2016 Yaroslav Akhremtsev <yaroslav.akhremtsev@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/

#pragma once

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "kahypar/datastructure/hash_table.h"
#include "kahypar/definitions.h"
#include "kahypar/utils/hash_vector.h"
#include "kahypar/utils/stats.h"

namespace kahypar {
template <typename _HashPolicy>
class AdaptiveLSHWithConnectedComponents {
 private:
  using HashPolicy = _HashPolicy;
  using BaseHashPolicy = typename HashPolicy::BaseHashPolicy;
  using HashValue = typename HashPolicy::HashValue;
  using MyHashSet = HashStorage<HashValue>;
  using Buckets = HashBuckets<HashValue, HypernodeID>;

 public:
  explicit AdaptiveLSHWithConnectedComponents(const Hypergraph& hypergraph,
                                              const Configuration& config) :
    _hypergraph(hypergraph),
    _config(config) { }

  std::vector<HypernodeID> build() {
    std::default_random_engine eng(_config.partition.seed);
    std::uniform_int_distribution<uint32_t> rnd;

    MyHashSet main_hash_set(0, _hypergraph.currentNumNodes());
    main_hash_set.reserve(20);

    std::vector<HypernodeID> clusters;
    clusters.reserve(_hypergraph.currentNumNodes());

    std::vector<uint32_t> cluster_size(_hypergraph.currentNumNodes(), 1);

    std::vector<uint8_t> active_clusters_bool_set(_hypergraph.currentNumNodes(), true);
    uint32_t num_active_vertices = _hypergraph.currentNumNodes();

    for (HypernodeID vertex_id = 0; vertex_id < _hypergraph.currentNumNodes(); ++vertex_id) {
      clusters.push_back(vertex_id);
    }

    std::vector<HypernodeID> inactive_clusters;
    inactive_clusters.reserve(_hypergraph.currentNumNodes());

    std::vector<HypernodeID> active_vertices_set;
    active_vertices_set.reserve(num_active_vertices);

    while (num_active_vertices > 0 && main_hash_set.getHashNum() <
           _config.preprocessing.min_hash_sparsifier.num_hash_functions) {
      main_hash_set.addHashVector();
      active_vertices_set.clear();

      for (HypernodeID vertex_id = 0; vertex_id < _hypergraph.currentNumNodes(); ++vertex_id) {
        HypernodeID cluster = clusters[vertex_id];
        if (active_clusters_bool_set[cluster]) {
          active_vertices_set.push_back(vertex_id);
        }
      }

      const uint32_t hash_num = main_hash_set.getHashNum() - 1;
      auto start = std::chrono::high_resolution_clock::now();
      incrementalParametersEstimation(active_vertices_set, rnd(eng), main_hash_set, hash_num);
      auto end = std::chrono::high_resolution_clock::now();
      Stats::instance().addToTotal(_config.partition.collect_stats,
                                   "adaptive_lsh_incremental_parameter_estimation",
                                   std::chrono::duration<double>(end - start).count());

      Buckets buckets(1, _hypergraph.currentNumNodes());
      start = std::chrono::high_resolution_clock::now();
      calculateOneDimBucket(_hypergraph.currentNumNodes(), active_clusters_bool_set, clusters,
                            main_hash_set, buckets, hash_num);
      end = std::chrono::high_resolution_clock::now();
      Stats::instance().addToTotal(_config.partition.collect_stats,
                                   "adaptive_lsh_construction_of_buckets",
                                   std::chrono::duration<double>(end - start).count());

      start = std::chrono::high_resolution_clock::now();
      calculateClustersIncrementally(_hypergraph.currentNumNodes(), active_clusters_bool_set,
                                     clusters, cluster_size,
                                     main_hash_set, hash_num, buckets, inactive_clusters);
      end = std::chrono::high_resolution_clock::now();
      Stats::instance().addToTotal(_config.partition.collect_stats,
                                   "adaptive_lsh_construction_of_clustering",
                                   std::chrono::duration<double>(end - start).count());

      std::vector<char> bit_map(clusters.size());
      for (const auto clst : clusters) {
        bit_map[clst] = 1;
      }

      size_t num_cl = 0;
      for (const auto bit : bit_map) {
        if (bit) {
          ++num_cl;
        }
      }
      LOG("Num clusters: " << num_cl);

      std::sort(inactive_clusters.begin(), inactive_clusters.end());
      auto end_iter = std::unique(inactive_clusters.begin(), inactive_clusters.end());
      inactive_clusters.resize(end_iter - inactive_clusters.begin());

      for (auto cluster : inactive_clusters) {
        active_clusters_bool_set[cluster] = false;
        num_active_vertices -= cluster_size[cluster];
      }
      inactive_clusters.clear();

      if (num_cl <= _hypergraph.currentNumNodes() / 2) {
        LOG("Adaptively chosen number of hash functions: " << main_hash_set.getHashNum());
        break;
      }
    }

    return clusters;
  }

 private:
  void incrementalParametersEstimation(std::vector<HypernodeID>& active_vertices,
                                       const uint32_t seed, MyHashSet& main_hash_set,
                                       const uint32_t main_hash_num) {
    MyHashSet hash_set(0, _hypergraph.currentNumNodes());
    hash_set.reserve(_config.preprocessing.min_hash_sparsifier.combined_num_hash_functions);

    BaseHashPolicy base_hash_policy(0, seed);
    base_hash_policy.reserveHashFunctions(
      _config.preprocessing.min_hash_sparsifier.combined_num_hash_functions);

    std::default_random_engine eng(seed);
    std::uniform_int_distribution<uint32_t> rnd;

    // in the beginning all vertices are in the same bucket (same hash)
    std::vector<HashValue> hashes(_hypergraph.currentNumNodes());

    size_t remained_vertices = active_vertices.size();

    const size_t buckets_num = _hypergraph.currentNumNodes();

    using Pair = std::pair<HashValue, HypernodeID>;

    std::vector<Pair> buckets;
    buckets.reserve(buckets_num);

    // empirically best value
    const uint32_t min_hash_num = 10;

    const uint32_t max_bucket_size = _config.preprocessing.min_hash_sparsifier.max_cluster_size;
    ASSERT(min_hash_num <= _config.preprocessing.min_hash_sparsifier.combined_num_hash_functions,
           "# min combined hash funcs should be <= # max combined hash funcs");
    std::vector<Pair> new_buckets;
    new_buckets.reserve(buckets_num);

    for (size_t i = 0; i + 1 < min_hash_num; ++i) {
      hash_set.addHashVector();
      base_hash_policy.addHashFunction(rnd(eng));

      base_hash_policy.calculateLastHash(_hypergraph, active_vertices, hash_set);

      const uint32_t last_hash = hash_set.getHashNum() - 1;
      for (const auto ver : active_vertices) {
        hashes[ver] ^= hash_set[last_hash][ver];
      }
    }

    for (const auto ver : active_vertices) {
      buckets.emplace_back(hashes[ver], ver);
    }
    std::sort(buckets.begin(), buckets.end());

    while (remained_vertices > 0) {
      hash_set.addHashVector();
      base_hash_policy.addHashFunction(rnd(eng));

      const uint32_t last_hash = hash_set.getHashNum() - 1;

      // Decide for which vertices we continue to increase the number of hash functions
      new_buckets.clear();

      auto begin = buckets.begin();

      while (begin != buckets.end()) {
        auto end = begin;
        while (end != buckets.end() && begin->first == end->first) {
          ++end;
        }
        std::vector<HypernodeID> vertices;
        vertices.reserve(end - begin);
        for (auto it = begin; it != end; ++it) {
          vertices.push_back(it->second);
        }

        base_hash_policy.calculateLastHash(_hypergraph, vertices, hash_set);

        if (vertices.size() == 1) {
          --remained_vertices;
          const HashValue hash = hash_set[last_hash][vertices.front()];
          hashes[vertices.front()] ^= hash;
          begin = end;
          continue;
        }

        std::vector<Pair> new_hashes;
        new_hashes.reserve(vertices.size());

        for (auto vertex : vertices) {
          const HashValue hash = hash_set[last_hash][vertex];
          hashes[vertex] ^= hash;

          new_hashes.emplace_back(hashes[vertex], vertex);
        }

        std::sort(new_hashes.begin(), new_hashes.end());
        const auto e = std::unique(new_hashes.begin(), new_hashes.end());
        new_hashes.resize(e - new_hashes.begin());

        auto new_begin = new_hashes.begin();
        while (new_begin != new_hashes.end()) {
          auto new_end = new_begin;
          while (new_end != new_hashes.end() && new_begin->first == new_end->first) {
            ++new_end;
          }
          const HypernodeID bucket_size = new_end - new_begin;
          if ((bucket_size <= max_bucket_size && hash_set.getHashNum() >= min_hash_num) ||
              hash_set.getHashNum() >=
              _config.preprocessing.min_hash_sparsifier.combined_num_hash_functions
              ) {
            remained_vertices -= bucket_size;
          } else {
            std::copy(new_begin, new_end, std::back_inserter(new_buckets));
          }
          new_begin = new_end;
        }
        begin = end;
      }
      buckets.swap(new_buckets);
    }
    for (uint32_t vertex_id = 0; vertex_id < _hypergraph.currentNumNodes(); ++vertex_id) {
      main_hash_set[main_hash_num][vertex_id] = hashes[vertex_id];
    }
  }

  // distributes vertices among bucket according to the new hash values
  void calculateOneDimBucket(const HypernodeID num_vertex,
                             const std::vector<uint8_t>& active_clusters_bool_set,
                             const std::vector<HypernodeID>& clusters, const MyHashSet& hash_set,
                             Buckets& buckets, const uint32_t hash_num) {
    ASSERT(buckets.isOneDimensional(), "Bucket should be one dimensional");
    const uint32_t dim = 0;

    for (HypernodeID vertex_id = 0; vertex_id < num_vertex; ++vertex_id) {
      const HypernodeID cluster = clusters[vertex_id];
      if (active_clusters_bool_set[cluster]) {
        buckets.put(dim, hash_set[hash_num][vertex_id], vertex_id);
      }
    }
  }

  // incrementally calculates clusters according to the new bucket
  void calculateClustersIncrementally(const HypernodeID num_vertex,
                                      std::vector<uint8_t>& active_clusters_bool_set,
                                      std::vector<HypernodeID>& clusters,
                                      std::vector<uint32_t>& cluster_size,
                                      const MyHashSet& hash_set, const uint32_t hash_num,
                                      Buckets& buckets,
                                      std::vector<HypernodeID>& inactive_clusters) {
    // Calculate clusters according to connected components
    // of an implicit graph induced by buckets. We use BFS on this graph.
    runIncrementalBfs(num_vertex, active_clusters_bool_set, hash_set, buckets, hash_num,
                      clusters, cluster_size, inactive_clusters);
  }

  void runIncrementalBfs(const HypernodeID num_vertex, std::vector<uint8_t>& active_clusters_bool_set,
                         const MyHashSet& hash_set, Buckets& buckets, const uint32_t hash_num,
                         std::vector<HypernodeID>& clusters, std::vector<uint32_t>& cluster_size,
                         std::vector<HypernodeID>& inactive_clusters) {
    std::queue<HypernodeID> vertex_queue;
    std::vector<char> visited(num_vertex, false);
    for (HypernodeID vertex_id = 0; vertex_id < num_vertex; ++vertex_id) {
      const HypernodeID cluster = clusters[vertex_id];
      if (!visited[vertex_id] && active_clusters_bool_set[cluster]) {
        visited[vertex_id] = true;
        vertex_queue.push(vertex_id);
        runIncrementalBfs(vertex_id, active_clusters_bool_set, hash_set, buckets, visited,
                          hash_num, clusters, cluster_size, inactive_clusters);
      }
    }
  }

  void runIncrementalBfs(const HypernodeID cur_vertex, std::vector<uint8_t>& active_clusters_bool_set,
                         const MyHashSet& hash_set,
                         Buckets& buckets, std::vector<char>& visited,
                         const uint32_t hash_num, std::vector<HypernodeID>& clusters,
                         std::vector<uint32_t>& cluster_size,
                         std::vector<HypernodeID>& inactive_clusters) {
    const HypernodeID cur_cluster = clusters[cur_vertex];
    const HashValue hash = hash_set[hash_num][cur_vertex];
    const uint32_t dim = buckets.isOneDimensional() ? 0 : hash_num;

    std::vector<HypernodeID> neighbours;
    neighbours.reserve(_config.preprocessing.min_hash_sparsifier.max_hyperedge_size);

    const auto iters = buckets.getObjects(dim, hash);
    for (auto neighbour_iter = iters.first; neighbour_iter != iters.second; ++neighbour_iter) {
      const auto neighbour = *neighbour_iter;
      const HypernodeID neighbour_cluster = clusters[neighbour];
      if (active_clusters_bool_set[neighbour_cluster]) {
        const HypernodeWeight weight = _hypergraph.nodeWeight(neighbour);
        if (cluster_size[cur_cluster] + weight >
            _config.preprocessing.min_hash_sparsifier.max_cluster_size) {
          ASSERT(cluster_size[cur_cluster] <=
                 _config.preprocessing.min_hash_sparsifier.max_cluster_size,
                 "Cluster is overflowed");
          break;
        }

        cluster_size[neighbour_cluster] -= weight;
        clusters[neighbour] = cur_cluster;
        cluster_size[cur_cluster] += weight;

        if (cluster_size[cur_cluster] >=
            _config.preprocessing.min_hash_sparsifier.min_cluster_size) {
          inactive_clusters.push_back(cur_cluster);
          active_clusters_bool_set[cur_cluster] = false;
        }

        visited[neighbour] = true;
        neighbours.push_back(neighbour);
      }
    }

    for (const auto neighbour : neighbours) {
      const uint32_t dim = buckets.isOneDimensional() ? 0 : hash_num;
      buckets.removeObject(dim, hash_set[hash_num][neighbour], neighbour);
    }
  }

  const Hypergraph& _hypergraph;
  const Configuration& _config;
};
}  // namespace kahypar