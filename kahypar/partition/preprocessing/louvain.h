/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2017 Sebastian Schlag <sebastian.schlag@kit.edu>
 * Copyright (C) 2017 Tobias Heuer <tobias.heuer@gmx.net>
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
#include <limits>
#include <vector>

#include "kahypar/datastructure/graph.h"
#include "kahypar/definitions.h"
#include "kahypar/macros.h"
#include "kahypar/meta/mandatory.h"
#include "kahypar/partition/configuration.h"
#include "kahypar/utils/randomize.h"

namespace kahypar {

template <class QualityMeasure = Mandatory,
          bool RandomizeNodes = true>
class Louvain {
 private:
  using Edge = ds::Edge;
  using Graph = ds::Graph;

 public:
  Louvain(const Hypergraph& hypergraph,
          const Configuration& config) :
    _graph_hierarchy(),
    _random_node_order(),
    _config(config) {
    _graph_hierarchy.emplace_back(hypergraph, config);
  }

  Louvain(const std::vector<NodeID>& adj_array,
          const std::vector<Edge>& edges,
          const Configuration& config) :
    _graph_hierarchy(),
    _random_node_order(),
    _config(config) {
    _graph_hierarchy.emplace_back(adj_array, edges);
  }

  EdgeWeight run() {
    bool improvement = false;
    size_t iteration = 0;
    EdgeWeight old_quality = -1.0L;
    EdgeWeight cur_quality = -1.0L;

    const size_t max_passes = std::numeric_limits<size_t>::max();


    std::vector<std::vector<NodeID> > mapping_stack;
    ASSERT(_graph_hierarchy.size() == 1);
    int cur_idx = 0;

    do {
      LOG("Graph Number Nodes: " << _graph_hierarchy[cur_idx].numNodes());
      LOG("Graph Number Edges: " << _graph_hierarchy[cur_idx].numEdges());
      QualityMeasure quality(_graph_hierarchy[cur_idx]);
      if (iteration == 0) {
        cur_quality = quality.quality();
      }

      ++iteration;
      LOG("######## Starting Louvain-Pass #" << iteration << " ########");

      //Checks if quality of the coarse graph is equal with the quality of next level finer graph
      ASSERT([&]() {
          if (cur_idx == 0) return true;
          if (std::abs(cur_quality - quality.quality()) > Graph::kEpsilon) {
            LOGVAR(cur_quality);
            LOGVAR(quality.quality());
            return false;
          }
          return true;
        } (), "Quality of contracted graph does not match quality of uncontracted graph");

      old_quality = cur_quality;
      HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
      cur_quality = louvain_pass(_graph_hierarchy[cur_idx], quality);
      HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      LOG("Louvain-Pass #" << iteration << " Time: " << elapsed_seconds.count() << "s");
      improvement = cur_quality - old_quality > _config.preprocessing.louvain_community_detection.min_eps_improvement;

      LOG("Louvain-Pass #" << iteration << " improve quality from " << old_quality
          << " to " << cur_quality);

      if (improvement) {
        cur_quality = quality.quality();
        LOG("Starting Contraction of communities...");
        start = std::chrono::high_resolution_clock::now();
        auto contraction = _graph_hierarchy[cur_idx++].contractClusters();
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        LOG("Contraction Time: " << elapsed_seconds.count() << "s");
        _graph_hierarchy.push_back(std::move(contraction.first));
        mapping_stack.push_back(std::move(contraction.second));
        LOG("Current number of communities: " << _graph_hierarchy[cur_idx].numCommunities());
      }

      LOG("");
    } while (improvement && iteration < max_passes);

    ASSERT((mapping_stack.size() + 1) == _graph_hierarchy.size());
    while (!mapping_stack.empty()) {
      assignClusterToNextLevelFinerGraph(_graph_hierarchy[cur_idx - 1], _graph_hierarchy[cur_idx],
                                         mapping_stack[cur_idx - 1]);
      _graph_hierarchy.pop_back();
      mapping_stack.pop_back();
      cur_idx--;
    }

    ASSERT(_graph_hierarchy.size() == 1);
    return cur_quality;
  }


  Graph getGraph() {
    return _graph_hierarchy[0];
  }

  ClusterID clusterID(const NodeID node) const {
    return _graph_hierarchy[0].clusterID(node);
  }

  ClusterID hypernodeClusterID(const HypernodeID hn) const {
    return _graph_hierarchy[0].hypernodeClusterID(hn);
  }

  ClusterID hyperedgeClusterID(const HyperedgeID he, const HypernodeID num_hns) const {
    return _graph_hierarchy[0].hyperedgeClusterID(he, num_hns);
  }

  size_t numCommunities() const {
    return _graph_hierarchy[0].numCommunities();
  }

 private:
  FRIEND_TEST(ALouvainAlgorithm, DoesOneLouvainPass);
  FRIEND_TEST(ALouvainAlgorithm, AssingsMappingToNextLevelFinerGraph);
  FRIEND_TEST(ALouvainKarateClub, DoesLouvainAlgorithm);

  void assignClusterToNextLevelFinerGraph(Graph& fine_graph, const Graph& coarse_graph,
                                          const std::vector<NodeID>& mapping) {
    for (const NodeID& node : fine_graph.nodes()) {
      fine_graph.setClusterID(node, coarse_graph.clusterID(mapping[node]));
    }
  }

  EdgeWeight louvain_pass(Graph& graph, QualityMeasure& quality) {
    size_t node_moves = 0;
    int iterations = 0;

    _random_node_order.clear();
    for (const NodeID& node : graph.nodes()) {
      _random_node_order.push_back(node);
    }

    // only false for testing purposes
    if (RandomizeNodes) {
      Randomize::instance().shuffleVector(_random_node_order, _random_node_order.size());
    }

    do {
      ++iterations;
      LOG("######## Starting Louvain-Pass-Iteration #" << iterations << " ########");
      node_moves = 0;
      for (const NodeID& node : _random_node_order) {
        const ClusterID cur_cid = graph.clusterID(node);
        EdgeWeight cur_incident_cluster_weight = 0.0L;
        ClusterID best_cid = cur_cid;
        EdgeWeight best_incident_cluster_weight = 0.0L;
        EdgeWeight best_gain = 0.0L;

        for (const Edge& e : graph.incidentEdges(node)) {
          if (graph.clusterID(e.target_node) == cur_cid && e.target_node != node) {
            cur_incident_cluster_weight += e.weight;
          }
        }
        best_incident_cluster_weight = cur_incident_cluster_weight;

        quality.remove(node, cur_incident_cluster_weight);

        for (const auto& cluster : graph.incidentClusterWeightOfNode(node)) {
          const ClusterID cid = cluster.clusterID;
          const EdgeWeight weight = cluster.weight;
          const EdgeWeight gain = quality.gain(node, cid, weight);
          if (gain > best_gain) {
            best_gain = gain;
            best_incident_cluster_weight = weight;
            best_cid = cid;
          }
        }

        quality.insert(node, best_cid, best_incident_cluster_weight);

        if (best_cid != cur_cid) {
          /*ASSERT([&]() {
          // Remove node from best cluster...
              quality.remove(node,best_incident_cluster_weight);
              // ... and insert in his old cluster.
              quality.insert(node,cur_cid,cur_incident_cluster_weight);
              EdgeWeight quality_before = quality.quality();
              //Remove node again from its old cluster ...
              quality.remove(node,cur_incident_cluster_weight);
              // ... and insert it in cluster with best gain.
              quality.insert(node,best_cid,best_incident_cluster_weight);
              EdgeWeight quality_after = quality.quality();
              if(quality_after - quality_before < -Graph::kEpsilon) {
                  LOGVAR(quality_before);
                  LOGVAR(quality_after);
                  return false;
              }
              return true;
          }(),"Move did not increase the quality!");*/

          ++node_moves;
        }
      }

      LOG("Iteration #" << iterations << ": Moving " << node_moves << " nodes to new communities.");
    } while (node_moves > 0 &&
             iterations < _config.preprocessing.louvain_community_detection.max_pass_iterations);


    return quality.quality();
  }

  std::vector<Graph> _graph_hierarchy;
  std::vector<NodeID> _random_node_order;
  const Configuration& _config;
};
}  // namespace kahypar
