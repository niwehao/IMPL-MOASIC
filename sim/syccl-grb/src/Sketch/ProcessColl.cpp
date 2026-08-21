// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include <thread>

void merge_chunk_set(std::map<chunk_t, int>& chunk_set_id, std::map<int, std::set<chunk_t>>& chunk_sets, std::set<chunk_t>& chunks, int& cur_set_id) {
  int set_id = -1;
  for (auto chunk : chunks) {
    if (chunk_set_id.find(chunk) == chunk_set_id.end()) {
      if (set_id == -1) set_id = cur_set_id++;
      chunk_set_id[chunk] = set_id;
      chunk_sets[set_id].insert(chunk);
    } else {
      if (set_id == -1) set_id = chunk_set_id[chunk];
      if (chunk_set_id[chunk] != set_id) {
        int prev_set_id = chunk_set_id[chunk];
        for (auto c : chunk_sets[prev_set_id]) {
          chunk_set_id[c] = set_id;
          chunk_sets[set_id].insert(c);
        }
        chunk_sets.erase(prev_set_id);
      }
    }
  }
}

int get_chunk_num(std::map<int, std::map<chunk_t, std::set<chunk_t>>> chunks) {
  int num = 0;
  for (const auto& it : chunks) {
    num += it.second.size();
  }
  return num;
}

std::vector<Collective*> extract_sub_collectives(Collective* coll) {
  LOOOG(ALGO, INFO, "Split collectives into independent sub-collectives ...");
  std::vector<Collective*> sub_colls;
  if (coll->coll_type == coll_allgather) {
    // into broadcast
    for (auto& [gpu_id, s_r_pair] : coll->sender_chunks) {
      std::set<chunk_t> chunk_sets;
      assert(s_r_pair.size() == 1);
      chunk_sets = s_r_pair.begin()->second;
      chunk_sets.insert(s_r_pair.begin()->first);
      sub_colls.push_back({new Collective(coll, chunk_sets)});
    }
  } else if (coll->coll_type == coll_alltoall) {
    // into scatter
    for (auto& [gpu_id, s_r_pair] : coll->sender_chunks) {
      std::set<chunk_t> chunk_sets;
      for (auto& [s, r] : s_r_pair) {
        chunk_sets.insert(s);
        for (auto& r1 : r)
          chunk_sets.insert(r1);
      }
      sub_colls.push_back({new Collective(coll, chunk_sets)});
    }
  } else {
    LOOOG(ALGO, INFO, "Unsupported collective", coll->coll_type);
    assert(0);
  }
  LOOOG(ALGO, INFO, "Extract %d sub-collectives", sub_colls.size());
  return sub_colls;
}

std::vector<Collective*> Algorithm::splitCollectives() {
  auto sub_colls = extract_sub_collectives(root_coll);
  std::vector<Collective*> isomorphic_coll_groups = {sub_colls[0]};
  return isomorphic_coll_groups;
}
