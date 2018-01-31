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
#include "kahypar/partition/context.h"
#include "kahypar/partition/metrics.h"
#include "kahypar/partition/refinement/2way_fm_gain_cache.h"
#include "kahypar/partition/refinement/fm_refiner_base.h"
#include "kahypar/partition/refinement/i_refiner.h"
#include "kahypar/partition/refinement/policies/fm_improvement_policy.h"
#include "kahypar/utils/float_compare.h"
#include "kahypar/utils/randomize.h"

namespace kahypar {

template <class StoppingPolicy = Mandatory,
          class FMImprovementPolicy = CutDecreasedOrInfeasibleImbalanceDecreased>
class TwoWayThSoftGainRefiner final : public IRefiner,
                              private FMRefinerBase<HypernodeID, FineGain>{
 private:
  static constexpr bool debug = false;

  using HypernodeWeightArray = std::array<HypernodeWeight, 2>;
  using Base = FMRefinerBase<HypernodeID, FineGain>;

 public:
  TwoWayThSoftGainRefiner(Hypergraph& hypergraph, const Context& context) :
    FMRefinerBase(hypergraph, context),
    _he_fully_active(_hg.initialNumEdges()),
    _hns_in_activation_vector(_hg.initialNumNodes()),
    _non_border_hns_to_remove(),
    _gain_cache(_hg.initialNumNodes()),
    _locked_hes(_hg.initialNumEdges(), HEState::free),
    _he_gain_contrib{TwoWayFMGainCache<double>(_hg.initialNumEdges()),
                     TwoWayFMGainCache<double>(_hg.initialNumEdges())},
    _stopping_policy() {
    ASSERT(context.partition.k == 2);
    _non_border_hns_to_remove.reserve(_hg.initialNumNodes());

    _thresholds = parseConfigString(context.local_search.fm.thresholds);
    _th_gain_factors = parseConfigString(context.local_search.fm.gain_factors);
  }

  ~TwoWayThSoftGainRefiner() override = default;

  TwoWayThSoftGainRefiner(const TwoWayThSoftGainRefiner&) = delete;
  TwoWayThSoftGainRefiner& operator= (const TwoWayThSoftGainRefiner&) = delete;

  TwoWayThSoftGainRefiner(TwoWayThSoftGainRefiner&&) = delete;
  TwoWayThSoftGainRefiner& operator= (TwoWayThSoftGainRefiner&&) = delete;

  void activate(const HypernodeID hn,
                const HypernodeWeightArray& max_allowed_part_weights) {
    if (_hg.isBorderNode(hn)) {
      ASSERT(!_hg.active(hn), V(hn));
      ASSERT(!_hg.marked(hn), V(hn));
      ASSERT(!_pq.contains(hn, 1 - _hg.partID(hn)), V(hn));
      ASSERT(ALMOST_EQUALS(_gain_cache.value(hn), computeGain(hn)), V(hn)
             << V(_gain_cache.value(hn)) << V(computeGain(hn)));

      DBG << "inserting HN" << hn << "with gain "
          << computeGain(hn) << "in PQ" << 1 - _hg.partID(hn);

      _pq.insert(hn, 1 - _hg.partID(hn), _gain_cache.value(hn));
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

    for (size_t sz = 0; sz < THF_CACHE_SIZE; sz++) {
      for (size_t t = 0; t < sz; t++) {
        _thf_cache[t][sz] = computeThresholdFactor(t, sz);
      }
    }

    for (int part = 0; part < 2; part++) {
      for (HyperedgeID he = 0; he < _hg.initialNumEdges(); he++) {
        _he_gain_contrib[part].setValue(he, 0);
      }
    }

    for (const HyperedgeID& he : _hg.edges()) {
      for (int part = 0; part < 2; part++) {
        _he_gain_contrib[part].setValue(he,
          _hg.pinCountInPart(he, 1-part) == _hg.edgeSize(he)
          ? 0
          : getThresholdFactor(_hg.pinCountInPart(he, 1-part), _hg.edgeSize(he)) * _hg.edgeWeight(he)
        );
      }
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
             FloatingPoint<double>(metrics::imbalance(_hg, _context))),
           V(best_metrics.imbalance) << V(metrics::imbalance(_hg, _context)));

    reset();
    _he_fully_active.reset();
    _locked_hes.resetUsedEntries();

    Randomize::instance().shuffleVector(refinement_nodes, refinement_nodes.size());
    for (const HypernodeID& hn : refinement_nodes) {
      for (const HyperedgeID& he : _hg.incidentEdges(hn)) {
        bool dirty = false;
        double thf_delta[2] = {0, 0};
        for (int part = 0; part < 2; part++) {
          double new_thf =
            _hg.pinCountInPart(he, 1-part) == _hg.edgeSize(he)
            ? 0
            : getThresholdFactor(_hg.pinCountInPart(he, 1-part), _hg.edgeSize(he))*_hg.edgeWeight(he);

          double old_thf = _he_gain_contrib[part].value(he);

          if (!ALMOST_EQUALS(old_thf, new_thf)) {
            dirty = true;
            thf_delta[part] = new_thf - old_thf;
            _he_gain_contrib[part].setValue(he, new_thf);
          }
        }

        for (const HypernodeID& pin : _hg.pins(he)) {
          if (_gain_cache.isCached(pin) && dirty) {
            _gain_cache.updateValue(pin, thf_delta[_hg.partID(pin)]);
          }
        }
      }
    }

    for (const HypernodeID& hn : refinement_nodes) {
      _gain_cache.setValue(hn, computeGain(hn));

      activate(hn, max_allowed_part_weights);

      // If Lmax0==Lmax1, then all border nodes should be active. However, if Lmax0 != Lmax1,
      // because k!=2^x or we intend to further partition the hypergraph into unequal number of
      // blocks, then it might not be possible to activate all refinement nodes, because a
      // part could be overweight regarding Lmax.
      ASSERT((_context.partition.max_part_weights[0] != _context.partition.max_part_weights[1]) ||
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
                                              _context, beta, best_metrics.cut, current_cut)) {
      ASSERT(_pq.isEnabled(0) || _pq.isEnabled(1));

      FineGain max_gain = kInvalidGain;
      HypernodeID max_gain_node = kInvalidHN;
      PartitionID to_part = Hypergraph::kInvalidPartition;

      _pq.deleteMax(max_gain_node, max_gain, to_part);
      // TODO(orlin): keep track of cuts separately
      Gain cut_delta = computeCut(max_gain_node);

      PartitionID from_part = _hg.partID(max_gain_node);

      ASSERT(!_hg.marked(max_gain_node), V(max_gain_node));
      ASSERT(_hg.isBorderNode(max_gain_node), V(max_gain_node));

      ASSERT(ALMOST_EQUALS(max_gain, computeGain(max_gain_node)));
      ASSERT(ALMOST_EQUALS(max_gain, _gain_cache.value(max_gain_node)));
      ASSERT([&]() {
          _hg.changeNodePart(max_gain_node, from_part, to_part);
          ASSERT((current_cut - cut_delta) == metrics::hyperedgeCut(_hg),
                 "cut=" << current_cut - cut_delta << "!=" << metrics::hyperedgeCut(_hg));
          _hg.changeNodePart(max_gain_node, to_part, from_part);
          return true;
        } ());

      DBG << V(current_cut) << V(max_gain_node) << V(max_gain) << V(from_part) << V(to_part)
          << V(_hg.nodeWeight(max_gain_node));

      _hg.changeNodePart(max_gain_node, from_part, to_part, _non_border_hns_to_remove);

      updatePQpartState(from_part, to_part, max_allowed_part_weights);

      current_imbalance = metrics::imbalance(_hg, _context);

      current_cut -= cut_delta;
      _stopping_policy.updateStatistics(cut_delta);

      ASSERT(current_cut == metrics::hyperedgeCut(_hg),
             V(current_cut) << V(metrics::hyperedgeCut(_hg)));
      _hg.mark(max_gain_node);
      updateNeighbours(max_gain_node, from_part, to_part, max_allowed_part_weights);

      _performed_moves.push_back(max_gain_node);

      // right now, we do not allow a decrease in cut in favor of an increase in balance
      const bool improved_cut_within_balance = (current_cut < best_metrics.cut) &&
                                               (_hg.partWeight(0)
                                                <= _context.partition.max_part_weights[0]) &&
                                               (_hg.partWeight(1)
                                                <= _context.partition.max_part_weights[1]);
      const bool improved_balance_less_equal_cut = (current_imbalance < best_metrics.imbalance) &&
                                                   (current_cut <= best_metrics.cut);
      const bool move_is_feasible = (_hg.partSize(from_part) > 0) &&
                                    (improved_cut_within_balance ||
                                     improved_balance_less_equal_cut);

      ++touched_hns_since_last_improvement;
      if (move_is_feasible) {
        DBGC(max_gain == 0) << "2WayFM improved balance between" << from_part << "and" << to_part
                            << "(max_gain=" << max_gain << ")";
        DBGC(current_cut < best_metrics.cut) << "2WayFM improved cut from" << best_metrics.cut
                                             << "to" << current_cut;
        best_metrics.cut = current_cut;
        best_metrics.imbalance = current_imbalance;
        _stopping_policy.resetStatistics();
        min_cut_index = _performed_moves.size() - 1;
        touched_hns_since_last_improvement = 0;
        _gain_cache.resetDelta();
        _he_gain_contrib[0].resetDelta();
        _he_gain_contrib[1].resetDelta();
      }
    }

    DBG << "KWayFM performed" << _performed_moves.size()
        << "local search movements ( min_cut_index=" << min_cut_index << "): stopped because of "
        << (_stopping_policy.searchShouldStop(touched_hns_since_last_improvement, _context, beta,
                                          best_metrics.cut, current_cut)
        == true ? "policy" : "empty queue");

    rollback(_performed_moves.size() - 1, min_cut_index);
    _gain_cache.rollbackDelta();
    _he_gain_contrib[0].rollbackDelta();
    _he_gain_contrib[1].rollbackDelta();

    ASSERT(best_metrics.cut == metrics::hyperedgeCut(_hg));
    ASSERT(best_metrics.cut <= initial_cut, V(initial_cut) << V(best_metrics.cut));
    ASSERT(best_metrics.imbalance == metrics::imbalance(_hg, _context),
           V(best_metrics.imbalance) << V(metrics::imbalance(_hg, _context)));
    ASSERT_THAT_GAIN_CACHE_IS_VALID();

    // This currently cannot be guaranteed in case k!=2^x, because initial partitioner might create
    // a bipartition which is balanced regarding epsilon, but not regarding the targeted block
    // weights Lmax0, Lmax1.
    // ASSERT(_hg.partWeight(0) <= _context.partition.max_part_weights[0], "Block 0 imbalanced");
    // ASSERT(_hg.partWeight(1) <= _context.partition.max_part_weights[1], "Block 1 imbalanced");
    return FMImprovementPolicy::improvementFound(best_metrics.cut, initial_cut,
                                                 best_metrics.imbalance,
                                                 initial_imbalance, _context.partition.epsilon);
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
    const FineGain old_gain = _gain_cache.value(moved_hn);
    const FineGain rb_delta = _gain_cache.delta(moved_hn);
    _gain_cache.setNotCached(moved_hn);
    for (const HyperedgeID& he : _hg.incidentEdges(moved_hn)) {
      if (_locked_hes.get(he) != HEState::locked) {
        if (_locked_hes.get(he) == to_part) {
          // he is loose
          deltaUpdate(from_part, to_part, he);
          DBG << "HE" << he << "maintained state: loose";
        } else if (_locked_hes.get(he) == HEState::free) {
          // he is free.
          fullUpdate(from_part, to_part, he);
          _locked_hes.set(he, to_part);
          DBG << "HE" << he << "changed state: free -> loose";
        } else {
          // he is loose and becomes locked after the move
          fullUpdate(from_part, to_part, he);
          _locked_hes.uncheckedSet(he, HEState::locked);
          DBG << "HE" << he << "changed state: loose -> locked";
        }
      } else {
        // he is locked
        DBG << he << "is locked";
        // In case of 2-FM, nothing to do here except keeping the cache up to date
        deltaUpdate(from_part, to_part, he);
      }
    }

    // TODO(orlin): try to get rid of this
    FineGain new_gain = computeGain(moved_hn);
    _gain_cache.setValue(moved_hn, new_gain);
    _gain_cache.setDelta(moved_hn, rb_delta + old_gain - new_gain);
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
                ASSERT(ALMOST_EQUALS(_pq.key(pin, other_part), computeGain(pin)),
                       V(pin) << V(computeGain(pin)) << V(_pq.key(pin, other_part))
                              << V(_hg.partID(pin)) << V(other_part));
              } else if (!_hg.marked(pin)) {
                ASSERT(true == false, "HN" << pin << "not in PQ, but also not marked!");
              }
            }
            // Gain calculation needs to be consistent in cache
            ASSERT(!_gain_cache.isCached(pin)
                || ALMOST_EQUALS(_gain_cache.value(pin), computeGain(pin)),
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
    if (ALMOST_EQUALS(gain_delta, 0)) {
      return;
    }

    // Only _gain_cache[moved_hn] = kNotCached, all other entries are cached.
    // However we set _gain_cache[moved_hn] to the correct value after all neighbors
    // are updated.
    _gain_cache.updateCacheAndDelta(pin, gain_delta);
  }

  void performNonZeroFullUpdate(const HypernodeID pin, const FineGain gain_delta,
                                HypernodeID& num_active_pins) KAHYPAR_ATTRIBUTE_ALWAYS_INLINE {
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

    HypernodeID num_active_pins = 1;  // because moved_hn was active
    HypernodeID sz = _hg.edgeSize(he);
    HypernodeID to_pins = _hg.pinCountInPart(he, to_part);
    HypernodeID from_pins = _hg.pinCountInPart(he, from_part);

    ASSERT(to_pins > 0);
    ASSERT(from_pins < sz);

    // if we're on the 'from' side, the gain depends on the # of pins on the 'to' side & vice-versa
    FineGain from_before = _he_gain_contrib[from_part].value(he);
    FineGain from_after = to_pins == sz ? 0 : getThresholdFactor(to_pins, sz) * _hg.edgeWeight(he);
    FineGain to_before = _he_gain_contrib[to_part].value(he);
    FineGain to_after = getThresholdFactor(from_pins, sz) * _hg.edgeWeight(he);

    ASSERT(ALMOST_EQUALS(_he_gain_contrib[from_part].value(he),
                         getThresholdFactor(to_pins-1, sz) * _hg.edgeWeight(he)),
        V(he) << V(_he_gain_contrib[from_part].value(he))
        << V(getThresholdFactor(to_pins-1, sz) * _hg.edgeWeight(he))
        << V(to_pins) << V(sz));

    ASSERT(from_pins+1 == sz ||
        ALMOST_EQUALS(_he_gain_contrib[to_part].value(he),
                      getThresholdFactor(from_pins+1, sz) * _hg.edgeWeight(he)),
        V(_he_gain_contrib[to_part].value(he))
        << V(getThresholdFactor(from_pins+1, sz) * _hg.edgeWeight(he)));

    ASSERT(from_pins+1 != sz || _he_gain_contrib[to_part].value(he) == 0,
        V(_he_gain_contrib[to_part].value(he)));

    for (const HypernodeID& pin : _hg.pins(he)) {
      FineGain& before = _hg.partID(pin) == from_part ? from_before : to_before;
      FineGain& after = _hg.partID(pin) == from_part ? from_after : to_after;

      performNonZeroFullUpdate(pin, after - before, num_active_pins);
    }

    _he_fully_active.set(he, (_hg.edgeSize(he) == num_active_pins));

    _he_gain_contrib[from_part].updateCacheAndDelta(he, from_after - from_before);
    _he_gain_contrib[to_part].updateCacheAndDelta(he, to_after - to_before);
  }

  // Delta-Gain Update as decribed in [ParMar06].
  // Removal of new non-border HNs is performed lazily after all updates
  // Used in the following cases:
  // - State transition: loose -> loose
  //   In this case, deltaUpdate<true> is called, since we perform
  //   a delta update induced by a local search move and we do want to update the PQ.
  // - State transition: locked -> locked
  //   In this case, we call deltaUpdate<false>, since we do not
  //   update the pq for locked HEs since locked HEs cannot be removed from the cut.
  template <bool update_local_search_pq = true>
  void deltaUpdate(const PartitionID from_part,
                   const PartitionID to_part, const HyperedgeID he) {

    HypernodeID sz = _hg.edgeSize(he);
    HypernodeID to_pins = _hg.pinCountInPart(he, to_part);
    HypernodeID from_pins = _hg.pinCountInPart(he, from_part);

    ASSERT(to_pins > 0);
    ASSERT(from_pins < sz);

    // if we're on the 'from' side, the gain depends on the # of pins on the 'to' side & vice-versa
    FineGain from_before = _he_gain_contrib[from_part].value(he);
    FineGain from_after = to_pins == sz ? 0 : getThresholdFactor(to_pins, sz) * _hg.edgeWeight(he);
    FineGain to_before = _he_gain_contrib[to_part].value(he);
    FineGain to_after = getThresholdFactor(from_pins, sz) * _hg.edgeWeight(he);

    ASSERT(ALMOST_EQUALS(_he_gain_contrib[from_part].value(he),
                         getThresholdFactor(to_pins-1, sz) * _hg.edgeWeight(he)),
        V(he) << V(_he_gain_contrib[from_part].value(he))
        << V(getThresholdFactor(to_pins-1, sz) * _hg.edgeWeight(he))
        << V(to_pins) << V(sz));

    ASSERT(from_pins+1 == sz ||
        ALMOST_EQUALS(_he_gain_contrib[to_part].value(he),
                      getThresholdFactor(from_pins+1, sz) * _hg.edgeWeight(he)),
        V(_he_gain_contrib[to_part].value(he))
        << V(getThresholdFactor(from_pins+1, sz) * _hg.edgeWeight(he)));

    ASSERT(from_pins+1 != sz || _he_gain_contrib[to_part].value(he) == 0,
        V(_he_gain_contrib[to_part].value(he)));

    FineGain from_delta = from_after - from_before;
    FineGain to_delta = to_after - to_before;

    for (const HypernodeID& pin : _hg.pins(he)) {
      const FineGain& delta = _hg.partID(pin) == from_part ? from_delta : to_delta;

      if (!_hg.marked(pin)) {
        updatePin(pin, delta);
        continue;
      }
      updateGainCache(pin, delta);
    }

    _he_gain_contrib[from_part].updateCacheAndDelta(he, from_delta);
    _he_gain_contrib[to_part].updateCacheAndDelta(he, to_delta);
  }

  void updatePin(const HypernodeID pin, const FineGain gain_delta) KAHYPAR_ATTRIBUTE_ALWAYS_INLINE {
    if (ALMOST_EQUALS(gain_delta, 0)) {
      return;
    }

    const PartitionID target_part = 1 - _hg.partID(pin);
    ASSERT(_hg.active(pin), V(pin) << V(target_part));
    ASSERT(_pq.contains(pin, target_part), V(pin) << V(target_part));
    ASSERT(!_hg.marked(pin));
    ASSERT(_gain_cache.isCached(pin), V(pin));

    DBG << "TwoWayThSoftGain updating gain of HN" << pin
        << "from gain" << _pq.key(pin, target_part) << "to "
        << _pq.key(pin, target_part) + gain_delta << "in PQ" << target_part;

    _pq.updateKeyBy(pin, target_part, gain_delta);
    _gain_cache.updateCacheAndDelta(pin, gain_delta);
  }

  void rollback(int last_index, const int min_cut_index) {
    DBG << "min_cut_index=" << min_cut_index;
    DBG << "last_index=" << last_index;
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

      gain += _hg.edgeWeight(he)
                * getThresholdFactor(_hg.pinCountInPart(he, _hg.partID(hn) ^ 1), _hg.edgeSize(he));
    }
    return gain;
  }

  Gain computeCut(const HypernodeID hn) const {
    Gain gain = 0;
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

  FineGain computeThresholdFactor(HypernodeID t, HypernodeID sz) const {
    FineGain part = (FineGain)t / (sz-1);
    auto it = std::lower_bound(_thresholds.begin(), _thresholds.end(), part);
    int index = it - _thresholds.begin();

    ASSERT(index >= 0);
    ASSERT(index < _th_gain_factors.size());

    return _th_gain_factors[index];
  }

  FineGain getThresholdFactor(HypernodeID t, HypernodeID sz) const {
    if (t < THF_CACHE_SIZE && sz < THF_CACHE_SIZE) {
      return _thf_cache[t][sz];
    } else {
      return computeThresholdFactor(t, sz);
    }
  }

  std::vector<double> parseConfigString(const std::string& input) const {
    std::istringstream iss(input);
    std::vector<double> result;
    while (iss) {
      double d;
      iss >> d;
      result.push_back(d);
      iss.get();
    }
    return result;
  }

  void ASSERT_THAT_GAIN_CACHE_IS_VALID() {
    ASSERT([&]() {
        for (const HypernodeID& hn : _hg.nodes()) {
          if (_gain_cache.isCached(hn) && !ALMOST_EQUALS(_gain_cache.value(hn), computeGain(hn))) {
            LOG << V(hn);
            LOG << V(_gain_cache.value(hn));
            LOG << V(computeGain(hn));
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
  using Base::_context;
  using Base::_pq;
  using Base::_performed_moves;
  using Base::_hns_to_activate;

  static const size_t THF_CACHE_SIZE = 101;

  ds::FastResetFlagArray<> _he_fully_active;
  ds::FastResetFlagArray<> _hns_in_activation_vector;  // faster than using a SparseSet in this case
  std::vector<HypernodeID> _non_border_hns_to_remove;
  TwoWayFMGainCache<FineGain> _gain_cache;
  ds::FastResetArray<PartitionID> _locked_hes;
  StoppingPolicy _stopping_policy;
  std::vector<double> _thresholds;
  std::vector<FineGain> _th_gain_factors;
  TwoWayFMGainCache<double> _he_gain_contrib[2];
  double _thf_cache[THF_CACHE_SIZE][THF_CACHE_SIZE];
};
}                                   // namespace kahypar
