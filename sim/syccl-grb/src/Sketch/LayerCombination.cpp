// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/searchutils.hpp"
#include <algorithm>
#include <set>
#include "common/ctpl.h"

bool output_sched_layers(prune_policy_t& prune, const sched_t& e, Simulation::topo_info_t& topo_info) {
  assert(prune.max_comb_layer_num == -1 || e.per_layer_size.size() <= prune.max_comb_layer_num);

  for (auto layer : prune.must_contain_layers)
    if (e.per_layer_size.find(layer) == e.per_layer_size.end())
      return false;

  for (auto& [layer, config] : e.per_layer_size) {
    if (topo_info.layer_info[layer].groups.size() == 1)
      return true;
    for (auto& [other_layer, other_config]: e.per_layer_size) {
      if (other_layer != layer && topo_info.layer_info[layer].layer_topo_relation[other_layer] == Simulation::layer_cross)
        return true;
    }
  }
  return false;
}

bool prune_sched_layers(prune_policy_t& prune, const sched_t& e, Simulation::topo_info_t& topo_info) {
  for (auto layer : prune.ignored_layers)
    if (e.per_layer_size.find(layer) != e.per_layer_size.end())
      return true;
  if (prune.max_comb_layer_num == -1)
    return false;
  return e.per_layer_size.size() == prune.max_comb_layer_num;
}

sched_t update_sched_layers(const sched_t& e, int layer) {
  sched_t e1 = e;
  e1.per_layer_size[layer] = {};
  return e1;
}

void finish_sched_layer(sched_t& e, topo_t topo) {
  for (auto& [layer, conf] : e.per_layer_size) {
    for (int group = 0; group < topo->topo_info.layer_info[layer].groups.size(); group++) {
      for (auto &src: topo->topo_info.layer_info[layer].groups[group].connected_gpus) {
        e.topo.gpu_layer_group[src][layer] = group;
        e.topo.layer_group_src_dest[layer][group][src] = topo->topo_info.layer_info[layer].groups[group].connected_gpus;
      }
    }
  }
  for (auto& [layer, conf] : e.per_layer_size) {
    assert(!topo->topo_info.layer_info[layer].groups.empty() && !topo->topo_info.layer_info[layer].groups.begin()->connected_gpus.empty());
    for (auto& [lower_layer, lower_conf] : e.per_layer_size) {
      if (lower_layer >= layer)
        break;
      for (int lower_group = 0; lower_group < topo->topo_info.layer_info[lower_layer].groups.size(); lower_group++) {
        for (auto &src: topo->topo_info.layer_info[lower_layer].groups[lower_group].connected_gpus) {
          int group = e.topo.gpu_layer_group[src][layer];
          for (auto &dst: topo->topo_info.layer_info[lower_layer].groups[lower_group].connected_gpus) {
            e.topo.layer_group_src_dest[layer][group][src].erase(dst);
          }
        }
      }
    }
  }
}

std::vector<sched_t> search_sched_layers_func(prune_policy_t& p, topo_t topo, const sched_t& e0) {
  std::vector<sched_t> outputs;
  std::queue<sched_t> tmp_q({e0});
  // for 2^{num_layers} combinations, filter out all valid ones
  for (int i = 0; i < topo->topo_info.layer_info.size(); i++) {
    int n = tmp_q.size();
    for (int j = 0; j < n; j++) {
      auto e = tmp_q.front();
      tmp_q.pop();
      // Ignore this layer for further search
      tmp_q.push(e);
      auto e1 = update_sched_layers(e, i);
      if (output_sched_layers(p, e1, topo->topo_info)) {
        // A valid combination
        LOG(ALGO, INFO, "Combination generated : %s", e1.toStr().c_str());
        finish_sched_layer(e1, topo);
        outputs.push_back(e1);
      } else if (prune_sched_layers(p, e1, topo->topo_info)) {
        continue;
      } else {
        // Include this layer for further search
        tmp_q.push(e1);
      }
    }
  }
  return outputs;
}

std::vector<sched_t> search_sched_layers(prune_policy_t& p, topo_t topo, const sched_t& input) {
  LOG(ALGO, INFO, "Search layer combinations ...");
  auto outputs = search_sched_layers_func(p, topo, input);

  for (auto& output_e : outputs) {
    LOG(ALGO, TRACE, "Generate: %s", output_e.toStr().c_str());
  }
  LOG(ALGO, INFO, "%d generated\n", outputs.size());
  return outputs;
}
