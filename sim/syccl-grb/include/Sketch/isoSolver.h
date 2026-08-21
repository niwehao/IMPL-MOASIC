// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PerfModel/Topology.h"
#include "Sketch/CommGraph.h"
#include <mutex>

inline std::mutex vector_mutex1;

void check_isomorphic_demand_simple(std::shared_ptr<Simulation::Topology> topo, int layer, int id1, int id2_to_check, std::vector<std::pair<int, intra_group_sched*>>& groups,
                      std::vector<int>& non_isomorphic_id, std::vector<std::pair<int, demand_map_t>>& isomorphic_id, std::set<int> &used_layers, bool a2a, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_parent, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_rem_size);
