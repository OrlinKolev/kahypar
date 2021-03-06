/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2014 Sebastian Schlag <sebastian.schlag@kit.edu>
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

#include <limits>
#include <vector>

#include "kahypar/datastructure/bucket_queue.h"
#include "kahypar/datastructure/kway_priority_queue.h"
#include "kahypar/definitions.h"
#include "kahypar/partition/context.h"

namespace kahypar {
struct RollbackInfo {
  HypernodeID hn;
  PartitionID from_part;
  PartitionID to_part;
};

template <typename RollbackElement = Mandatory, typename GainType=Gain,
          typename GainLimits = std::numeric_limits<GainType> >
class FMRefinerBase {
 private:
  static constexpr bool debug = false;

 public:
  FMRefinerBase(const FMRefinerBase&) = delete;
  FMRefinerBase& operator= (const FMRefinerBase&) = delete;

  FMRefinerBase(FMRefinerBase&&) = delete;
  FMRefinerBase& operator= (FMRefinerBase&&) = delete;

  ~FMRefinerBase() = default;

 protected:
  static constexpr HypernodeID kInvalidHN = std::numeric_limits<HypernodeID>::max();
  static const GainType kInvalidGain;
  static constexpr HyperedgeWeight kInvalidDecrease = std::numeric_limits<PartitionID>::min();

  enum HEState {
    free = std::numeric_limits<PartitionID>::max() - 1,
    locked = std::numeric_limits<PartitionID>::max(),
  };

#ifdef USE_BUCKET_QUEUE
#warning Bucket queue not tested with non-integral gain type (may need to replace numeric_limits::min with ::lowest)
  using KWayRefinementPQ = ds::KWayPriorityQueue<HypernodeID, GainType,
                                                 std::numeric_limits<GainType>,
                                                 false,
                                                 ds::EnhancedBucketQueue<HypernodeID,
                                                                         GainType,
                                                                         std::numeric_limits<GainType>
                                                                         > >;
#else
  using KWayRefinementPQ = ds::KWayPriorityQueue<HypernodeID, GainType, GainLimits>;
#endif


  FMRefinerBase(Hypergraph& hypergraph, const Context& context) :
    _hg(hypergraph),
    _context(context),
    _pq(context.partition.k),
    _performed_moves(),
    _hns_to_activate() {
    _performed_moves.reserve(_hg.initialNumNodes());
    _hns_to_activate.reserve(_hg.initialNumNodes());
  }

  bool hypernodeIsConnectedToPart(const HypernodeID pin, const PartitionID part) const {
    for (const HyperedgeID& he : _hg.incidentEdges(pin)) {
      if (_hg.pinCountInPart(he, part) > 0) {
        return true;
      }
    }
    return false;
  }

  bool moveIsFeasible(const HypernodeID max_gain_node, const PartitionID from_part,
                      const PartitionID to_part) const {
    ASSERT(_context.partition.mode == Mode::direct_kway,
           "Method should only be called in direct partitioning");
    return (_hg.partWeight(to_part) + _hg.nodeWeight(max_gain_node)
            <= _context.partition.max_part_weights[0]) && (_hg.partSize(from_part) - 1 != 0);
  }

  void moveHypernode(const HypernodeID hn, const PartitionID from_part,
                     const PartitionID to_part) {
    ASSERT(_hg.isBorderNode(hn), "Hypernode" << hn << "is not a border node!");
    DBG << "moving HN" << hn << "from" << from_part
        << "to" << to_part << "(weight=" << _hg.nodeWeight(hn) << ")";
    _hg.changeNodePart(hn, from_part, to_part);
  }

  PartitionID heaviestPart() const {
    PartitionID heaviest_part = 0;
    for (PartitionID part = 1; part < _context.partition.k; ++part) {
      if (_hg.partWeight(part) > _hg.partWeight(heaviest_part)) {
        heaviest_part = part;
      }
    }
    return heaviest_part;
  }

  void reCalculateHeaviestPartAndItsWeight(PartitionID& heaviest_part,
                                           HypernodeWeight& heaviest_part_weight,
                                           const PartitionID from_part,
                                           const PartitionID to_part) const {
    if (heaviest_part == from_part) {
      heaviest_part = heaviestPart();
      heaviest_part_weight = _hg.partWeight(heaviest_part);
    } else if (_hg.partWeight(to_part) > heaviest_part_weight) {
      heaviest_part = to_part;
      heaviest_part_weight = _hg.partWeight(to_part);
    }
    ASSERT(heaviest_part_weight == _hg.partWeight(heaviestPart()),
           V(heaviest_part) << V(heaviestPart()));
  }

  void reset() {
    _pq.clear();
    _hg.resetHypernodeState();
    _performed_moves.clear();
  }

  void rollback(int last_index, const int min_cut_index) {
    DBG << "min_cut_index=" << min_cut_index;
    DBG << "last_index=" << last_index;
    while (last_index != min_cut_index) {
      const HypernodeID hn = _performed_moves[last_index].hn;
      const PartitionID from_part = _performed_moves[last_index].to_part;
      const PartitionID to_part = _performed_moves[last_index].from_part;
      _hg.changeNodePart(hn, from_part, to_part);
      --last_index;
    }
  }

  Hypergraph& _hg;
  const Context& _context;
  KWayRefinementPQ _pq;
  std::vector<RollbackElement> _performed_moves;
  std::vector<HypernodeID> _hns_to_activate;
};

template <typename RollbackElement, typename GainType, typename GainLimits>
const GainType FMRefinerBase<RollbackElement, GainType, GainLimits>::kInvalidGain
  = GainLimits::lowest();
}  // namespace kahypar
