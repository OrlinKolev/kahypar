/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2014-2016 Sebastian Schlag <sebastian.schlag@kit.edu>
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
#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "kahypar/datastructure/binary_heap.h"
#include "kahypar/datastructure/fast_reset_array.h"
#include "kahypar/datastructure/fast_reset_flag_array.h"
#include "kahypar/datastructure/sparse_set.h"
#include "kahypar/definitions.h"
#include "kahypar/meta/mandatory.h"
#include "kahypar/meta/template_parameter_to_string.h"
#include "kahypar/partition/configuration.h"
#include "kahypar/partition/metrics.h"
#include "kahypar/partition/refinement/2way_fm_gain_cache.h"
#include "kahypar/partition/refinement/fm_refiner_base.h"
#include "kahypar/partition/refinement/i_refiner.h"
#include "kahypar/partition/refinement/policies/2fm_rebalancing_policy.h"
#include "kahypar/partition/refinement/policies/fm_improvement_policy.h"
#include "kahypar/utils/float_compare.h"
#include "kahypar/utils/randomize.h"

namespace kahypar {
static const bool dbg_refinement_2way_netstatus_improvements_cut = false;
static const bool dbg_refinement_2way_netstatus_improvements_balance = false;
static const bool dbg_refinement_2way_netstatus_stopping_crit = false;
static const bool dbg_refinement_2way_netstatus_gain_update = false;
static const bool dbg_refinement_2way_netstatus__activation = false;

template <class StoppingPolicy = Mandatory,
          class UseGlobalRebalancing = NoGlobalRebalancing,
          class FMImprovementPolicy = CutDecreasedOrInfeasibleImbalanceDecreased>
class TwoWayNetstatusRefiner final : public IRefiner,
                              private FMRefinerBase<HypernodeID, FineGain>{
 private:
  using RebalancePQ = ds::BinaryMaxHeap<HypernodeID, FineGain>;
  using HypernodeWeightArray = std::array<HypernodeWeight, 2>;
  using Base = FMRefinerBase<HypernodeID, FineGain>;
  using TempGainMap = std::unordered_map<HypernodeID, FineGain>;

 public:
  TwoWayNetstatusRefiner(Hypergraph& hypergraph, const Configuration& config) :
    FMRefinerBase(hypergraph, config),
    _rebalance_pqs({ RebalancePQ(_hg.initialNumNodes()), RebalancePQ(_hg.initialNumNodes()) }),
    _he_fully_active(_hg.initialNumEdges()),
    _hns_in_activation_vector(_hg.initialNumNodes()),
    _non_border_hns_to_remove(),
    _gain_cache(_hg.initialNumNodes()),
    _locked_hes(_hg.initialNumEdges(), HEState::free),
    _stopping_policy() {
    ASSERT(config.partition.k == 2);
    _non_border_hns_to_remove.reserve(_hg.initialNumNodes());
  }

  ~TwoWayNetstatusRefiner() override = default;

  TwoWayNetstatusRefiner(const TwoWayNetstatusRefiner&) = delete;
  TwoWayNetstatusRefiner& operator= (const TwoWayNetstatusRefiner&) = delete;

  TwoWayNetstatusRefiner(TwoWayNetstatusRefiner&&) = delete;
  TwoWayNetstatusRefiner& operator= (TwoWayNetstatusRefiner&&) = delete;

  FineGain getLooseHEDelta(HyperedgeID he) {
    static const Gain MAX = _config.partition.hyperedge_size_threshold;
    FineGain size = _hg.pins(he).second - _hg.pins(he).first;
    return (MAX / size) * (_locked_pins[he] / (size - _locked_pins[he]));
  }

  void activate(const HypernodeID hn,
                const HypernodeWeightArray& max_allowed_part_weights) {
    if (_hg.isBorderNode(hn)) {
      ASSERT(!_hg.active(hn), V(hn));
      ASSERT(!_hg.marked(hn), V(hn));
      ASSERT(!_pq.contains(hn, 1 - _hg.partID(hn)), V(hn));
      ASSERT(_gain_cache.value(hn) == computeGain(hn), V(hn)
             << V(_gain_cache.value(hn)) << V(computeGain(hn)));

      _pq.insert(hn, 1 - _hg.partID(hn), _gain_cache.value(hn) + _temp_gains[hn]);
      if (_hg.partWeight(1 - _hg.partID(hn)) < max_allowed_part_weights[1 - _hg.partID(hn)]) {
        _pq.enablePart(1 - _hg.partID(hn));
      }
      _hg.activate(hn);
    }
  }

  bool isInitialized() const {
    return _is_initialized;
  }

 private:
  FRIEND_TEST(ATwoWayNetstatusRefiner, IdentifiesBorderHypernodes);
  FRIEND_TEST(ATwoWayNetstatusRefiner, ComputesPartitionSizesOfHE);
  FRIEND_TEST(ATwoWayNetstatusRefiner, ChecksIfPartitionSizesOfHEAreAlreadyCalculated);
  FRIEND_TEST(ATwoWayNetstatusRefiner, ComputesGainOfHypernodeMovement);
  FRIEND_TEST(ATwoWayNetstatusRefiner, ActivatesBorderNodes);
  FRIEND_TEST(ATwoWayNetstatusRefiner, CalculatesNodeCountsInBothPartitions);
  FRIEND_TEST(ATwoWayNetstatusRefiner, UpdatesNodeCountsOnNodeMovements);
  FRIEND_TEST(AGainUpdateMethod, RespectsPositiveGainUpdateSpecialCaseForHyperedgesOfSize2);
  FRIEND_TEST(AGainUpdateMethod, RespectsNegativeGainUpdateSpecialCaseForHyperedgesOfSize2);
  // TODO(schlag): find better names for testcases
  FRIEND_TEST(AGainUpdateMethod, HandlesCase0To1);
  FRIEND_TEST(AGainUpdateMethod, HandlesCase1To0);
  FRIEND_TEST(AGainUpdateMethod, HandlesCase2To1);
  FRIEND_TEST(AGainUpdateMethod, HandlesCase1To2);
  FRIEND_TEST(AGainUpdateMethod, HandlesSpecialCaseOfHyperedgeWith3Pins);
  FRIEND_TEST(AGainUpdateMethod, ActivatesUnmarkedNeighbors);
  FRIEND_TEST(AGainUpdateMethod, RemovesNonBorderNodesFromPQ);
  FRIEND_TEST(ATwoWayNetstatusRefiner, UpdatesPartitionWeightsOnRollBack);
  FRIEND_TEST(AGainUpdateMethod, DoesNotDeleteJustActivatedNodes);
  FRIEND_TEST(ARefiner, DoesNotDeleteMaxGainNodeInPQ0IfItChoosesToUseMaxGainNodeInPQ1);
  FRIEND_TEST(ARefiner, ChecksIfMovePreservesBalanceConstraint);
  FRIEND_TEST(ATwoWayNetstatusRefinerDeathTest, ConsidersSingleNodeHEsDuringInitialGainComputation);
  FRIEND_TEST(ATwoWayNetstatusRefiner, KnowsIfAHyperedgeIsFullyActive);

  void initializeImpl(const HyperedgeWeight max_gain) override final {
    if (!_is_initialized) {
#ifdef USE_BUCKET_QUEUE
      _pq.initialize(_hg.initialNumNodes(), max_gain);
#else
      unused(max_gain);
      _pq.initialize(_hg.initialNumNodes());
#endif
      _is_initialized = true;
    }
    _gain_cache.clear();
    for (const HypernodeID& hn : _hg.nodes()) {
      _gain_cache.setValue(hn, computeGain(hn));
      ASSERT(_gain_cache.value(hn) == computeGain(hn), V(hn)
             << V(_gain_cache.value(hn)) << V(computeGain(hn)));
    }
  }

  bool refineImpl(std::vector<HypernodeID>& refinement_nodes,
                  const HypernodeWeightArray& max_allowed_part_weights,
                  const UncontractionGainChanges& changes,
                  Metrics& best_metrics) override final {
    ASSERT(best_metrics.cut == metrics::hyperedgeCut(_hg),
           V(best_metrics.cut) << V(metrics::hyperedgeCut(_hg)));
    ASSERT(FloatingPoint<double>(best_metrics.imbalance).AlmostEquals(
             FloatingPoint<double>(metrics::imbalance(_hg, _config))),
           V(best_metrics.imbalance) << V(metrics::imbalance(_hg, _config)));

    reset();
    _he_fully_active.reset();
    _locked_hes.resetUsedEntries();
    _locked_pins.clear();
    _temp_gains.clear();

    _gain_cache.setValue(refinement_nodes[0], computeGain(refinement_nodes[0]));
    _gain_cache.setValue(refinement_nodes[1], computeGain(refinement_nodes[1]));

    Randomize::instance().shuffleVector(refinement_nodes, refinement_nodes.size());
    for (const HypernodeID& hn : refinement_nodes) {
      activate(hn, max_allowed_part_weights);

      // If Lmax0==Lmax1, then all border nodes should be active. However, if Lmax0 != Lmax1,
      // because k!=2^x or we intend to further partition the hypergraph into unequal number of
      // blocks, then it might not be possible to activate all refinement nodes, because a
      // part could be overweight regarding Lmax.
      ASSERT((_config.partition.max_part_weights[0] != _config.partition.max_part_weights[1]) ||
             (!_hg.isBorderNode(hn) || _pq.isEnabled(1 - _hg.partID(hn))), V(hn));
    }

    ASSERT_THAT_GAIN_CACHE_IS_VALID();

    const HyperedgeWeight initial_cut = best_metrics.cut;
    const double initial_imbalance = best_metrics.imbalance;
    HyperedgeWeight current_cut = best_metrics.cut;
    double current_imbalance = best_metrics.imbalance;

    int min_cut_index = -1;
    int touched_hns_since_last_improvement = 0;
    _stopping_policy.resetStatistics();

    const double beta = log(_hg.currentNumNodes());
    while (!_pq.empty() &&
           !_stopping_policy.searchShouldStop(touched_hns_since_last_improvement,
                                              _config, beta, best_metrics.cut, current_cut)) {
      ASSERT(_pq.isEnabled(0) || _pq.isEnabled(1));

      FineGain max_gain = kInvalidGain;
      HypernodeID max_gain_node = kInvalidHN;
      PartitionID to_part = Hypergraph::kInvalidPartition;

      _pq.deleteMax(max_gain_node, max_gain, to_part);

      PartitionID from_part = _hg.partID(max_gain_node);

      ASSERT(!_hg.marked(max_gain_node), V(max_gain_node));
      ASSERT(_hg.isBorderNode(max_gain_node), V(max_gain_node));

      ASSERT(ALMOST_EQUALS(max_gain, computeGain(max_gain_node) + _temp_gains[max_gain_node]));
      ASSERT(ALMOST_EQUALS(max_gain, _gain_cache.value(max_gain_node) + _temp_gains[max_gain_node]));
      ASSERT([&]() {
          _hg.changeNodePart(max_gain_node, from_part, to_part);
          ASSERT(ALMOST_EQUALS(
              (current_cut - (max_gain - _temp_gains[max_gain_node])),
              metrics::hyperedgeCut(_hg)),
                 "cut=" << current_cut - max_gain << "!=" << metrics::hyperedgeCut(_hg));
          _hg.changeNodePart(max_gain_node, to_part, from_part);
          return true;
        } ());

      _hg.changeNodePart(max_gain_node, from_part, to_part, _non_border_hns_to_remove);

      updatePQpartState(from_part, to_part, max_allowed_part_weights);

      current_imbalance = metrics::imbalance(_hg, _config);

      current_cut -= _gain_cache.value(max_gain_node);
      _stopping_policy.updateStatistics(max_gain - _temp_gains[max_gain_node]);

      ASSERT(current_cut == metrics::hyperedgeCut(_hg),
             V(current_cut) << V(metrics::hyperedgeCut(_hg)));
      _hg.mark(max_gain_node);
      updateNeighbours(max_gain_node, from_part, to_part, max_allowed_part_weights);

      _performed_moves.push_back(max_gain_node);

      // right now, we do not allow a decrease in cut in favor of an increase in balance
      const bool improved_cut_within_balance = (current_cut < best_metrics.cut) &&
                                               (_hg.partWeight(0)
                                                <= _config.partition.max_part_weights[0]) &&
                                               (_hg.partWeight(1)
                                                <= _config.partition.max_part_weights[1]);
      const bool improved_balance_less_equal_cut = (current_imbalance < best_metrics.imbalance) &&
                                                   (current_cut <= best_metrics.cut);
      const bool move_is_feasible = (_hg.partSize(from_part) > 0) &&
                                    (improved_cut_within_balance ||
                                     improved_balance_less_equal_cut);
      ++touched_hns_since_last_improvement;
      if (move_is_feasible) {
        best_metrics.cut = current_cut;
        best_metrics.imbalance = current_imbalance;
        _stopping_policy.resetStatistics();
        min_cut_index = _performed_moves.size() - 1;
        touched_hns_since_last_improvement = 0;
        _gain_cache.resetDelta();
      }
    }

    rollback(_performed_moves.size() - 1, min_cut_index);
    _gain_cache.rollbackDelta<UseGlobalRebalancing>(_rebalance_pqs, _hg);

    ASSERT(best_metrics.cut == metrics::hyperedgeCut(_hg));
    ASSERT(best_metrics.cut <= initial_cut, V(initial_cut) << V(best_metrics.cut));
    ASSERT(best_metrics.imbalance == metrics::imbalance(_hg, _config),
           V(best_metrics.imbalance) << V(metrics::imbalance(_hg, _config)));
    ASSERT_THAT_GAIN_CACHE_IS_VALID();

    // This currently cannot be guaranteed in case k!=2^x, because initial partitioner might create
    // a bipartition which is balanced regarding epsilon, but not regarding the targeted block
    // weights Lmax0, Lmax1.
    // ASSERT(_hg.partWeight(0) <= _config.partition.max_part_weights[0], "Block 0 imbalanced");
    // ASSERT(_hg.partWeight(1) <= _config.partition.max_part_weights[1], "Block 1 imbalanced");
    return FMImprovementPolicy::improvementFound(best_metrics.cut, initial_cut,
                                                 best_metrics.imbalance,
                                                 initial_imbalance, _config.partition.epsilon);
  }


  void updatePQpartState(const PartitionID from_part, const PartitionID to_part,
                         const HypernodeWeightArray& max_allowed_part_weights) {
    if (_hg.partWeight(to_part) >= max_allowed_part_weights[to_part]) {
      _pq.disablePart(to_part);
    }
    if (_hg.partWeight(from_part) < max_allowed_part_weights[from_part]) {
      _pq.enablePart(from_part);
    }
  }

  void removeInternalizedHns() {
    for (const HypernodeID& hn : _non_border_hns_to_remove) {
      // The just moved HN might be contained in the vector since changeNodePart
      // does not explicitly check for that HN. However it may still
      // be a border node - but it is marked as moved for sure.
      // All other HNs contained in the vector have to be internal nodes.
      ASSERT(_hg.marked(hn) || !_hg.isBorderNode(hn), V(hn));
      if (_hg.active(hn)) {
        ASSERT(_pq.contains(hn, (1 - _hg.partID(hn))), V(hn) << V((1 - _hg.partID(hn))));
        _pq.remove(hn, (_hg.partID(hn) ^ 1));
        _hg.deactivate(hn);
      }
    }
    _non_border_hns_to_remove.clear();
  }
  
  void updateNeighbours(const HypernodeID moved_hn, const PartitionID from_part,
                        const PartitionID to_part,
                        const HypernodeWeightArray& max_allowed_part_weights) {
    const FineGain temp = _gain_cache.value(moved_hn);
    ASSERT(-temp == computeGain(moved_hn), V(moved_hn));
    const FineGain rb_delta = _gain_cache.delta(moved_hn);
    _gain_cache.setNotCached(moved_hn);
    for (const HyperedgeID& he : _hg.incidentEdges(moved_hn)) {
      if (_locked_hes.get(he) != HEState::locked) {
        if (_locked_hes.get(he) == to_part) {
          // he is loose
          deltaUpdate(from_part, to_part, he);
        } else if (_locked_hes.get(he) == HEState::free) {
          // he is free.
          fullUpdate(from_part, to_part, he);
          _locked_hes.set(he, to_part);
        } else {
          // he is loose and becomes locked after the move
          fullUpdate(from_part, to_part, he);
          _locked_hes.uncheckedSet(he, HEState::locked);
        }
      } else {
        // he is locked
        // In case of 2-FM, nothing to do here except keeping the cache up to date
        deltaUpdate<  /*rebalacing update */ false,  /*update pq */ false>(from_part, to_part, he);
      }
    }

    _gain_cache.setValue(moved_hn, -temp);
    _gain_cache.setDelta(moved_hn, rb_delta + 2 * temp);

    for (const HypernodeID& hn : _hns_to_activate) {
      ASSERT(!_hg.active(hn), V(hn));
      activate(hn, max_allowed_part_weights);
    }
    _hns_to_activate.clear();
    _hns_in_activation_vector.reset();

    // changeNodePart collects all pins that become non-border hns after the move
    // Previously, these nodes were removed directly in fullUpdate. While this is
    // certainly the correct place to do so, it lead to a significant overhead, because
    // each time we looked at at pin, it was necessary to check whether or not it still
    // is a border hypernode. By delaying the removal until all updates are performed
    // (and therefore doing some unnecessary updates) we get rid of the border node check
    // in fullUpdate, which significantly reduces the running time for large hypergraphs like
    // kkt_power.
    removeInternalizedHns();

    ASSERT([&]() {
        for (const HyperedgeID& he : _hg.incidentEdges(moved_hn)) {
          for (const HypernodeID& pin : _hg.pins(he)) {
            const PartitionID other_part = 1 - _hg.partID(pin);
            if (!_hg.isBorderNode(pin)) {
              // The pin is an internal HN
              ASSERT(!_pq.contains(pin, other_part), V(pin));
              ASSERT(!_hg.active(pin), V(pin));
            } else {
              // HN is border HN
              // Border HNs should be contained in PQ or be marked
              ASSERT(!_hg.active(pin) || _pq.contains(pin, other_part), V(pin));
              if (_pq.contains(pin, other_part)) {
                ASSERT(!_hg.marked(pin), V(pin));
                ASSERT(ALMOST_EQUALS(_pq.key(pin, other_part), computeGain(pin) + _temp_gains[pin]),
                       V(pin) << V(computeGain(pin)) << V(_temp_gains[pin])
                       << V(_pq.key(pin, other_part)) << V(_hg.partID(pin)) << V(other_part));
              } else if (!_hg.marked(pin)) {
                ASSERT(true == false, "HN " << pin << " not in PQ, but also not marked!");
              }
            }
            // Gain calculation needs to be consistent in cache
            ASSERT(!_gain_cache.isCached(pin) || _gain_cache.value(pin) == computeGain(pin),
                   V(pin) << V(_gain_cache.value(pin)) << V(computeGain(pin)));
          }
        }
        return true;
      } (), "UpdateNeighbors failed!");

    ASSERT((!_pq.empty(0) && _hg.partWeight(0) < max_allowed_part_weights[0] ?
            _pq.isEnabled(0) : !_pq.isEnabled(0)), V(0));
    ASSERT((!_pq.empty(1) && _hg.partWeight(1) < max_allowed_part_weights[1] ?
            _pq.isEnabled(1) : !_pq.isEnabled(1)), V(1));
  }

  void updateGainCache(const HypernodeID pin, const FineGain gain_delta) KAHYPAR_ATTRIBUTE_ALWAYS_INLINE {
    // Only _gain_cache[moved_hn] = kNotCached, all other entries are cached.
    // However we set _gain_cache[moved_hn] to the correct value after all neighbors
    // are updated.
    _gain_cache.updateCacheAndDelta(pin, gain_delta);
  }

  void performNonZeroFullUpdate(const HypernodeID pin, const FineGain gain_delta,
                                HypernodeID& num_active_pins) KAHYPAR_ATTRIBUTE_ALWAYS_INLINE {
    ASSERT(gain_delta != 0);
    if (!_hg.marked(pin)) {
      if (!_hg.active(pin)) {
        if (!_hns_in_activation_vector[pin]) {
          ASSERT(!_pq.contains(pin, (1 - _hg.partID(pin))), V(pin) << V((1 - _hg.partID(pin))));
          ++num_active_pins;  // since we do lazy activation!
          _hns_to_activate.push_back(pin);
          _hns_in_activation_vector.set(pin, true);
        }
      } else {
        updatePin(pin, gain_delta);
        ++num_active_pins;
        return;    // caching is done in updatePin in this case
      }
    }
    updateGainCache(pin, gain_delta);
  }

  // Full update includes:
  // 1.) Activation of new border HNs (lazy)
  // 2.) Delta-Gain Update as decribed in [ParMar06].
  // Removal of new non-border HNs is performed lazily after all updates
  // This is used for the state transitions: free -> loose and loose -> locked
  void fullUpdate(const PartitionID from_part,
                  const PartitionID to_part, const HyperedgeID he) {
    
    FineGain state_gain = 0;
    if (_locked_hes.get(he) == HEState::free) {
      _locked_pins[he] = 1;
      state_gain += getLooseHEDelta(he);
    } else {
      state_gain -= getLooseHEDelta(he);
    }

    for (const HypernodeID& pin : _hg.pins(he)) {
      _temp_gains[pin] += state_gain;
      const PartitionID target_part = 1 - _hg.partID(pin);
      if (_pq.contains(pin)) {
        _pq.updateKeyBy(pin, target_part, state_gain);
      }
    }

    const HypernodeID pin_count_from_part_after_move = _hg.pinCountInPart(he, from_part);
    const HypernodeID pin_count_to_part_after_move = _hg.pinCountInPart(he, to_part);

    const bool he_became_cut_he = pin_count_to_part_after_move == 1;
    const bool he_became_internal_he = pin_count_from_part_after_move == 0;
    const bool increase_necessary = pin_count_from_part_after_move == 1;
    const bool decrease_necessary = pin_count_to_part_after_move == 2;

    if (he_became_cut_he || he_became_internal_he || increase_necessary ||
        decrease_necessary || !_he_fully_active[he]) {
      ASSERT(_hg.edgeSize(he) != 1, V(he));
      const HyperedgeWeight he_weight = _hg.edgeWeight(he);
      HypernodeID num_active_pins = 1;  // because moved_hn was active

      if (_hg.edgeSize(he) == 2) {
        for (const HypernodeID& pin : _hg.pins(he)) {
          performNonZeroFullUpdate(pin, (_hg.partID(pin) == from_part ? 2 : -2) * he_weight,
                                   num_active_pins);
        }
      } else if (he_became_cut_he) {
        // HE was an internal edge before move and is a cut HE now.
        // Before the move, all pins had gain -w(e). Now after the move,
        // these pins have gain 0 (since all of them are in from_part).
        for (const HypernodeID& pin : _hg.pins(he)) {
          performNonZeroFullUpdate(pin, he_weight, num_active_pins);
        }
      } else if (he_became_internal_he) {
        // HE was cut HE before move and is internal HE now.
        // Since the HE is now internal, moving a pin incurs gain -w(e)
        // and make it a cut HE again.
        for (const HypernodeID& pin : _hg.pins(he)) {
          performNonZeroFullUpdate(pin, -he_weight, num_active_pins);
        }
      } else {
        for (const HypernodeID& pin : _hg.pins(he)) {
          // factor is unfortunately necessary because we need to iterate over all pins
          // since we night find new nodes for activation.

          // Before move, there were two pins (moved_node and the current pin) in from_part.
          // After moving moved_node to to_part, the gain of the remaining pin in
          // from_part increases by w(he).
          FineGain factor = increase_necessary && _hg.partID(pin) == from_part ? 1 : 0;
          // Before move, pin was the only HN in to_part. It thus had a
          // positive gain, because moving it to from_part would have removed
          // the HE from the cut. Now, after the move, pin becomes a 0-gain HN
          // because now there are pins in both parts.
          factor = decrease_necessary && _hg.partID(pin) != from_part ? -1 : factor;
          if (!_hg.marked(pin)) {
            if (!_hg.active(pin)) {
              if (!_hns_in_activation_vector[pin]) {
                ASSERT(!_pq.contains(pin, (1 - _hg.partID(pin))), V(pin) << V((1 - _hg.partID(pin))));
                ++num_active_pins;  // since we do lazy activation!
                _hns_to_activate.push_back(pin);
                _hns_in_activation_vector.set(pin, true);
              }
            } else {
              if (factor != 0) {
                updatePin(pin, factor * he_weight);
              }
              ++num_active_pins;
              continue;    // caching is done in updatePin in this case
            }
          }
          if (factor != 0) {
            updateGainCache(pin, factor * he_weight);
          }
        }
      }
      _he_fully_active.set(he, (_hg.edgeSize(he) == num_active_pins));
    }
  }

  // Delta-Gain Update as decribed in [ParMar06].
  // Removal of new non-border HNs is performed lazily after all updates
  // Used in the following cases:
  // - State transition: loose -> loose
  //   In this case, deltaUpdate<false,true> is called, since we perform
  //   a delta update induced by a local search move (and thus not a rebalancing
  //   move) and we do want to update the PQ.
  // - State transition: locked -> locked
  //   In this case, we call deltaUpdate<false,false>, since we do not
  //   update the pq for locked HEs since locked HEs cannot be removed from the cut.
  // - Update because of rebalancing move
  //   In this case, we call deltaUpdate<true,true> because the delta update is
  //   due to a rebalancing  move. In this case we have to check for active nodes
  //   (first template parameter) and want to update the PQ (second template parameter).
  template <bool is_rebalancing_update = false,
            bool update_local_search_pq = true>
  void deltaUpdate(const PartitionID from_part,
                   const PartitionID to_part, const HyperedgeID he) {

    if (update_local_search_pq) {
      FineGain state_gain = -getLooseHEDelta(he);
      _locked_pins[he]++;
      state_gain += getLooseHEDelta(he);

      for (const HypernodeID& pin : _hg.pins(he)) {
        _temp_gains[pin] += state_gain;
        const PartitionID target_part = 1 - _hg.partID(pin);
        if (_pq.contains(pin)) {
          _pq.updateKeyBy(pin, target_part, state_gain);
        }
      }
    }

    const HypernodeID pin_count_from_part_after_move = _hg.pinCountInPart(he, from_part);
    const HypernodeID pin_count_to_part_after_move = _hg.pinCountInPart(he, to_part);

    const bool he_became_cut_he = pin_count_to_part_after_move == 1;
    const bool he_became_internal_he = pin_count_from_part_after_move == 0;
    const bool increase_necessary = pin_count_from_part_after_move == 1;
    const bool decrease_necessary = pin_count_to_part_after_move == 2;

    if (he_became_cut_he || he_became_internal_he || increase_necessary ||
        decrease_necessary) {
      ASSERT(_hg.edgeSize(he) != 1, V(he));
      const HyperedgeWeight he_weight = _hg.edgeWeight(he);

      if (_hg.edgeSize(he) == 2) {
        for (const HypernodeID& pin : _hg.pins(he)) {
          const char factor = (_hg.partID(pin) == from_part ? 2 : -2);
          if (update_local_search_pq && !_hg.marked(pin)) {
            updatePin(pin, factor * he_weight);
            continue;      // caching is done in updatePin in this case
          }
          updateGainCache(pin, factor * he_weight);
        }
      } else if (he_became_cut_he) {
        for (const HypernodeID& pin : _hg.pins(he)) {
          if (update_local_search_pq && !_hg.marked(pin)) {
            updatePin(pin, he_weight);
            continue;      // caching is done in updatePin in this case
          }
          updateGainCache(pin, he_weight);
        }
      } else if (he_became_internal_he) {
        for (const HypernodeID& pin : _hg.pins(he)) {
          if (update_local_search_pq && !_hg.marked(pin)) {
            updatePin(pin, -he_weight);
            continue;      // caching is done in updatePin in this case
          }
          updateGainCache(pin, -he_weight);
        }
      } else {
        for (const HypernodeID& pin : _hg.pins(he)) {
          if (_hg.partID(pin) == from_part) {
            if (increase_necessary) {
              if (update_local_search_pq && !_hg.marked(pin)) {
                updatePin(pin, he_weight);
                // break;      // caching is done in updatePin in this case
              } else {
                updateGainCache(pin, he_weight);
              }
            }
          } else if (decrease_necessary) {
            if (update_local_search_pq && !_hg.marked(pin)) {
              updatePin(pin, -he_weight);
              // break;    // caching is done in updatePin in this case
            } else {
              updateGainCache(pin, -he_weight);
            }
          }
        }
      }
    }
  }

  std::string policyStringImpl() const override final {
    return std::string(" RefinerStoppingPolicy=" + meta::templateToString<StoppingPolicy>() +
                       " RefinerUsesBucketQueue=" +
#ifdef USE_BUCKET_PQ
                       "true");
#else
                       "false");
#endif
  }

  void updatePin(const HypernodeID pin, const FineGain gain_delta) KAHYPAR_ATTRIBUTE_ALWAYS_INLINE {
    const PartitionID target_part = 1 - _hg.partID(pin);
    ASSERT(_hg.active(pin), V(pin) << V(target_part));
    ASSERT(_pq.contains(pin, target_part), V(pin) << V(target_part));
    ASSERT(gain_delta != 0, V(gain_delta));
    ASSERT(!_hg.marked(pin));
    ASSERT(_gain_cache.isCached(pin), V(pin));

    _pq.updateKeyBy(pin, target_part, gain_delta);
    _gain_cache.updateCacheAndDelta(pin, gain_delta);
  }

  void rollback(int last_index, const int min_cut_index) {
    while (last_index != min_cut_index) {
      HypernodeID hn = _performed_moves[last_index];
      _hg.changeNodePart(hn, _hg.partID(hn), (_hg.partID(hn) ^ 1));
      --last_index;
    }
  }

  FineGain computeGain(const HypernodeID hn) const {
    FineGain gain = 0;
    ASSERT(_hg.partID(hn) < 2);
    for (const HyperedgeID& he : _hg.incidentEdges(hn)) {
      ASSERT(_hg.edgeSize(he) > 1, V(he));
      if (_hg.pinCountInPart(he, _hg.partID(hn) ^ 1) == 0) {
        gain -= _hg.edgeWeight(he);
      }
      if (_hg.pinCountInPart(he, _hg.partID(hn)) == 1) {
        gain += _hg.edgeWeight(he);
      }
    }
    return gain;
  }

  void ASSERT_THAT_GAIN_CACHE_IS_VALID() {
    ASSERT([&]() {
        for (const HypernodeID& hn : _hg.nodes()) {
          if (_gain_cache.isCached(hn) && _gain_cache.value(hn) != computeGain(hn)) {
            LOGVAR(hn);
            LOGVAR(_gain_cache.value(hn));
            LOGVAR(computeGain(hn));
            return false;
          }
        }
        return true;
      } (), "GainCache Invalid");
  }

  bool ALMOST_EQUALS(FineGain a, FineGain b) {
    return std::abs(a-b) < 1e-6;
  }

  using Base::_hg;
  using Base::_config;
  using Base::_pq;
  using Base::_performed_moves;
  using Base::_hns_to_activate;

  std::array<RebalancePQ, 2> _rebalance_pqs;
  ds::FastResetFlagArray<> _he_fully_active;
  ds::FastResetFlagArray<> _hns_in_activation_vector;  // faster than using a SparseSet in this case
  std::vector<HypernodeID> _non_border_hns_to_remove;
  TwoWayFMGainCache<FineGain> _gain_cache;
  ds::FastResetArray<PartitionID> _locked_hes;
  StoppingPolicy _stopping_policy;
  TempGainMap _temp_gains;
  std::unordered_map<HyperedgeID, unsigned int> _locked_pins;
};
}                                   // namespace kahypar
