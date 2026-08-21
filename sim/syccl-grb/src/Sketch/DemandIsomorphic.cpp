// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/CommGraph.h"
#include "Sketch/isoSolver.h"
#include "Sketch/common.h"

// For alltoall, additionally consider remaining sizes(routing info between srcs and dsts)
std::pair<bool, demand_map_t> check_shift_constraint_demand(Simulation::topo_info_t &topo_info, ExtractedLayerInfo &info, int layer, intra_group_sched& s1, intra_group_sched& s2, int g1, int g2, std::map<int, std::vector<int>>* cp1, std::map<int, std::vector<int>>* cp2, std::map<int, std::vector<int>>* rs1, std::map<int, std::vector<int>>* rs2, int offset, std::set<int> &used_layers, bool a2a) {
  // LOOOG(ALGO, INFO, "Checking shift constraint, layer %d, g1 %d, g2 %d, offset %d", layer, g1, g2, offset);
  std::vector<int> c1, c2;
  for (auto &[chunk_id, src_dsts] : s1.src_dest_pairs) {
    c1.push_back(chunk_id);
  }
  for (auto &[chunk_id, src_dsts] : s2.src_dest_pairs) {
    c2.push_back(chunk_id);
  }
  if (c1.size() != c2.size()) return {false, {}};
  auto &layer_info = topo_info.layer_info[layer];
  auto &g1_info = layer_info.groups[g1], &g2_info = layer_info.groups[g2];
  if (g1_info.type != g2_info.type) return {false, {}};
  // local chunk_id -> local src/dst pairs
  std::map<int, std::pair<std::set<std::tuple<int, int, int>>, std::set<std::tuple<int, int, int>>>> loc1, loc2;
  std::vector<int> chunk_local_to_global1, chunk_local_to_global2;
  for (auto &[chunk_id, src_dsts] : s1.src_dest_pairs) {
    auto &[srcs, dsts] = src_dsts;
    // For allgather, we only need the first element in srcs_loc and dsts_loc
    std::set<std::tuple<int, int, int>> srcs_loc, dsts_loc;
    for (auto src: srcs) {
      srcs_loc.insert({g1_info.global_to_local_ids.at(src), -1, 0});
    }
    // Compared to allgather, we additonally require the a2a_parent and the amount of transferred from the parent for every dst to be the same after mapping, so we additionally record these fields
    for (auto dst: dsts) {
      if (a2a) {
        auto p_global = (*cp1)[chunk_id][dst];
        auto p_local = g1_info.global_to_local_ids.at(p_global);
        auto rem = (*rs1)[chunk_id][dst];
        dsts_loc.insert({g1_info.global_to_local_ids.at(dst), p_local, rem});  
      } else {
        dsts_loc.insert({g1_info.global_to_local_ids.at(dst), -1, 0});  
      }
    }
    int local_id = chunk_local_to_global1.size();
    chunk_local_to_global1.push_back(chunk_id);
    loc1[local_id] = {srcs_loc, dsts_loc};
  }
  for (auto &[chunk_id, src_dsts] : s2.src_dest_pairs) {
    auto &[srcs, dsts] = src_dsts;
    std::set<std::tuple<int, int, int>> srcs_loc, dsts_loc;
    for (auto src: srcs) {
      srcs_loc.insert({g2_info.global_to_local_ids.at(src), -1, 0});
    }
    for (auto dst: dsts) {
      if (a2a) {
        auto p_global = (*cp2)[chunk_id][dst];
        auto p_local = g2_info.global_to_local_ids.at(p_global);
        auto rem = (*rs2)[chunk_id][dst];
        dsts_loc.insert({g2_info.global_to_local_ids.at(dst), p_local, rem});  
      } else {
        dsts_loc.insert({g2_info.global_to_local_ids.at(dst), -1, 0});  
      }
    }
    int local_id = chunk_local_to_global2.size();
    chunk_local_to_global2.push_back(chunk_id);
    loc2[local_id] = {srcs_loc, dsts_loc};
  }
  std::map<int, std::pair<std::set<std::tuple<int, int, int>>, std::set<std::tuple<int, int, int>>>> loc1_shifted;
  // TODO: WARNING: this only checks for switched topologies
  // As stated in the mapping generation, some topologies with certain layer partition may not be compatible and the mapping is not guarenteed to preserve the topology
  // So other mappings are required for that
  int nchunks = c1.size();
  GpuMapping mapping = genSingleMapping(offset, info, used_layers);
  std::set<int> used_gpus_global;
  for (int i = 0; i < nchunks; ++i) {
    try {
      auto &pair = loc1[i];
      auto &[srcs, dsts] = pair;
      std::set<std::tuple<int, int, int>> srcs_shifted, dsts_shifted;
      for (auto [src, p, rem]: srcs) {
        auto src_global = g1_info.local_to_global_ids.at(src);
        used_gpus_global.insert(src_global);
        auto mapped_global = mapping.node_map[src_global];
        auto mapped_local = g1_info.global_to_local_ids.at(mapped_global);
        srcs_shifted.insert({mapped_local, p, rem});
      }
      for (auto [dst, p, rem]: dsts) {
        auto dst_global = g1_info.local_to_global_ids.at(dst);
        used_gpus_global.insert(dst_global);
        auto mapped_global = mapping.node_map[dst_global];
        auto mapped_local = g1_info.global_to_local_ids.at(mapped_global);
        if (a2a){
          auto p_global = g1_info.local_to_global_ids.at(p);
          auto p_mapped_global = mapping.node_map[p_global];
          auto p_mapped_local = g1_info.global_to_local_ids.at(p_mapped_global);
          dsts_shifted.insert({mapped_local, p_mapped_local, rem});  
        } else {
          dsts_shifted.insert({mapped_local, -1, 0});  
        }
      }
      loc1_shifted[i] = {srcs_shifted, dsts_shifted};
    } catch (std::exception &e) {
      // Some offset naturally doesn't work for certain groups as the offset are global offsets
      return {false, {}};
    }
  }
  std::map<int, int> chunk_mapping;
  BipartiteGraph graph(nchunks, nchunks);
  for (int i = 0; i < nchunks; ++i) {
    for (int j = 0; j < nchunks; ++j) {
      if (loc1_shifted[i] == loc2[j]) {
        graph.addEdge(i + 1, j + 1);
      }
    }
  }
  int found = graph.hopcroftKarp();
  if (found) {
    auto matches = graph.getMatchesFromU();
    for (int i = 1; i <= nchunks; ++i) {
      chunk_mapping[i - 1] = matches[i] - 1;
    }
  }
  demand_map_t ret;
  if (found == nchunks) {
    /*
    std::lock_guard<std::mutex> guard(log_mutex);
    LOOOG(ALGO, INFO, "found isomorphic demand between %s and %s", s1.toStr().c_str(), s2.toStr().c_str());
    LOOOG(ALGO, INFO, "g1_info.local_to_global_ids=%s", mapToStr(g1_info.local_to_global_ids).c_str());
    LOOOG(ALGO, INFO, "g2_info.local_to_global_ids=%s", mapToStr(g2_info.local_to_global_ids).c_str());
    LOOOG(ALGO, INFO, "chunk_mapping=%s", mapToStr(chunk_mapping).c_str());
    LOOOG(ALGO, INFO, "node_map=%s", mapToStr(mapping.node_map).c_str());
    LOOOG(ALGO, INFO, "used_gpus_global=%s", setToStr(used_gpus_global).c_str());
    */
    auto &global_chunk_mapping = ret.chunk_map;
    for (auto &[a, b] : chunk_mapping) {
      global_chunk_mapping[chunk_local_to_global1[a]] = chunk_local_to_global2[b];
    }
    auto &global_gpu_mapping = ret.gpu_map;
    for (auto &[src, dst] : mapping.node_map) {
      if (!used_gpus_global.count(src)) continue;
      auto mapped_global = mapping.node_map[src];
      auto mapped_local = g1_info.global_to_local_ids.at(mapped_global);
      auto image_global = g2_info.local_to_global_ids.at(mapped_local);
      global_gpu_mapping[src] = image_global;
    }
    // LOOOG(ALGO, INFO, "mapping=%s", ret.toStr().c_str());
  }
  return {found == nchunks, ret};
}

void check_isomorphic_demand_simple(std::shared_ptr<Simulation::Topology> topo, int layer, int id1, int id2_to_check, std::vector<std::pair<int, intra_group_sched*>>& groups,
                      std::vector<int>& non_isomorphic_id, std::vector<std::pair<int, demand_map_t>>& isomorphic_id, std::set<int> &used_layers, bool a2a, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_parent, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_rem_size) {
  // LOOOG(ALGO, INFO, "checking isomorphic demand between %d and %d", id1, id2_to_check);
  auto [g1, s1] = groups[id1];
  auto [g2, s2] = groups[id2_to_check];
  auto a2a_chunk_parent1 = sched_a2a_chunk_parent[s1], a2a_chunk_parent2 = sched_a2a_chunk_parent[s2];
  auto a2a_chunk_rem_size1 = sched_a2a_chunk_rem_size[s1], a2a_chunk_rem_size2 = sched_a2a_chunk_rem_size[s2];
  auto &d1 = *s1, &d2 = *s2;
  int sz1 = d1.src_dest_pairs.size(), sz2 = d2.src_dest_pairs.size();
  bool iso = false;
  demand_map_t mapping;
  if (sz1 != sz2) {
    iso = false;
  } else {
    ExtractedLayerInfo info(topo->topo_info, used_layers);
    int n = info.group_size[layer][0];
    // tryout all possible shifts inside this group
    for (int i = 0; i < n; ++i) {
      auto [pass, map] = check_shift_constraint_demand(topo->topo_info, info, layer, d1, d2, g1, g2, a2a_chunk_parent1, a2a_chunk_parent2, a2a_chunk_rem_size1, a2a_chunk_rem_size2, i, used_layers, a2a);
      if (pass) {
        mapping = map;
        iso = true;
        break;
      }
    }
  }

  // Re-assure the mapping is valid
  if (iso) {
    std::lock_guard<std::mutex> guard(vector_mutex1);
    isomorphic_id.emplace_back(id2_to_check, mapping);
    LOOOG(ALGO, TRACE, "%s is isomorphic to %s via mapping %s", d1.toStr().c_str(), d2.toStr().c_str(), isomorphic_id.back().second.toStr().c_str());
    auto &p1 = d1.src_dest_pairs, &p2 = d2.src_dest_pairs;
    // check if the mapping holds
    for (auto &[chunk_id, src_dsts] : p1) {
      auto &[srcs, dsts] = src_dsts;
      auto &[srcs2, dsts2] = p2[mapping.chunk_map[chunk_id]];
      auto &acp1 = (*a2a_chunk_parent1)[chunk_id], &acp2 = (*a2a_chunk_parent2)[mapping.chunk_map[chunk_id]];
      auto &ars1 = (*a2a_chunk_rem_size1)[chunk_id], &ars2 = (*a2a_chunk_rem_size2)[mapping.chunk_map[chunk_id]];
      assert(srcs.size() == srcs2.size());
      assert(dsts.size() == dsts2.size());
      std::set<int> mapped_srcs, mapped_dsts;
      for (auto src: srcs) {
        mapped_srcs.insert(mapping.gpu_map[src]);
      }
      for (auto dst: dsts) {
        mapped_dsts.insert(mapping.gpu_map[dst]);
        if (a2a){
          // Additonally check a2a routing info
          assert(mapping.gpu_map[acp1[dst]] == acp2[mapping.gpu_map[dst]]);
          assert(ars1[dst] == ars2[mapping.gpu_map[dst]]);
        }
      }
      assert(mapped_srcs == srcs2);
      assert(mapped_dsts == dsts2);
    }
    if (!isomorphic_id.back().second.checkValid()) {
      LOOOG(ALGO, INFO, "invalid mapping %s", isomorphic_id.back().second.toStr().c_str());
      assert(false);
    }
  } else {
    std::lock_guard<std::mutex> guard(vector_mutex1);
    non_isomorphic_id.push_back(id2_to_check);
    LOOOG(ALGO, TRACE, "%s is not isomorphic to %s", d1.toStr().c_str(), d2.toStr().c_str());
    for (auto &[chunk, parents] : *a2a_chunk_parent1) {
      LOOOG(ALGO, TRACE, "cp1: chunk %d=%s", chunk, vecToStr(parents).c_str());
    }
    for (auto &[chunk, parents] : *a2a_chunk_parent2) {
      LOOOG(ALGO, TRACE, "cp2: chunk %d=%s", chunk, vecToStr(parents).c_str());
    }
    for (auto &[chunk, rem_size] : *a2a_chunk_rem_size1) {
      LOOOG(ALGO, TRACE, "rs1: chunk %d=%s", chunk, vecToStr(rem_size).c_str());
    }
    for (auto &[chunk, rem_size] : *a2a_chunk_rem_size2) {
      LOOOG(ALGO, TRACE, "rs2: chunk %d=%s", chunk, vecToStr(rem_size).c_str());
    }
  }
}