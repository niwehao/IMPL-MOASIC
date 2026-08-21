// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/searchutils.hpp"
#include <algorithm>
#include <set>
#include "common/ctpl.h"

int layer_num_groups(const topo_subset_t& topo, int layer) {
  assert(topo.layer_group_src_dest.find(layer) != topo.layer_group_src_dest.end());
  return topo.layer_group_src_dest.at(layer).size();
}

int layer_num_gpus_per_group(const topo_subset_t& topo, int layer) {
  assert(topo.layer_group_src_dest.find(layer) != topo.layer_group_src_dest.end());
  assert(!topo.layer_group_src_dest.at(layer).empty());
  auto g = topo.layer_group_src_dest.at(layer).begin();
  //return topo.layer_group_src_dest.at(layer).begin()->second.size();
  return g->second.size();
}

int layer_num_links_per_gpu(const topo_subset_t& topo, int layer) {
  assert(topo.layer_group_src_dest.find(layer) != topo.layer_group_src_dest.end());
  assert(!topo.layer_group_src_dest.at(layer).empty());
  assert(!topo.layer_group_src_dest.at(layer).begin()->second.empty());
  return topo.layer_group_src_dest.at(layer).begin()->second.begin()->second.size();
}

int layer_comm(const sched_layer_config_t& config) {
  int res = 0;
  for (auto& conf : config.config)
    res += conf.dest_per_group * conf.used_group_num;
  return res;
}


bool output_sched_size(prune_policy_t& policy, const sched_t& e) {
  bool at_least_one_layer_no_source_with_0_dest = !policy.at_least_one_layer_no_source_with_0_dest;
  for (auto& [layer, config] : e.per_layer_size) {
    for (auto group_config: config.config) {
      if (group_config.non_comm_source_per_group == 0)
        at_least_one_layer_no_source_with_0_dest = true;
    }
  }
  if (!at_least_one_layer_no_source_with_0_dest)
    return false;


  for (auto& [layer, config] : e.per_layer_size) {
    if (policy.layer_comm_uplimit.find(layer) != policy.layer_comm_uplimit.end()) {
      if (policy.layer_comm_uplimit.at(layer) < layer_comm(e.per_layer_size.at(layer)))
        return false;
    }
    for (auto& [other_layer, other_config] : e.per_layer_size) {
      if (policy.layer_comm_ordering[layer].find(other_layer) != policy.layer_comm_ordering[layer].end()) {
        if (layer_comm(e.per_layer_size.at(layer)) < layer_comm(e.per_layer_size.at(other_layer)))
          return false;
      }
    }
  }

  for (auto& [layer, config] : e.per_layer_size) {
    int group_num = layer_num_groups(e.topo, layer);
    bool is_all_group_used = group_num == total_num_groups(e, layer);
    bool one_group_config_used = config.config.size() == 1;
    bool source_group_special = false;
    for (auto& c : config.config) {
      if (c.group_id == e.topo.gpu_layer_group.at(e.sender).at(layer))
        source_group_special = true;
    }
    if (is_all_group_used && (one_group_config_used || source_group_special))
      return true;
  }

  for (auto& [layer, config] : e.per_layer_size) {
    int group_num = layer_num_groups(e.topo, layer);
    bool is_all_group_used = group_num == total_num_groups(e, layer);
    bool one_group_config_used = config.config.size() == 1;
    if (is_all_group_used && one_group_config_used)
      return true;
  }
  return false;
}

bool sched_finish(const sched_t& e) {
  for (auto& [layer, config] : e.per_layer_size) {
    if (config.config.empty()) {
      return false;
    }
  }
  return true;
}

void get_remained_dest(const sched_t& e, int& max, int& min) {
  max = 0;
  min = 0;
  for (auto& [layer, layer_comm] : e.per_layer_size) {
    max += layer_num_groups(e.topo, layer) * (layer_num_gpus_per_group(e.topo, layer) -1);
    min += 1;
  }
}

void update_remained_dest(int group_num, int group_gpu_num, int& max, int& min) {
  max -= group_num*(group_gpu_num-1);
  min -= 1;
}

std::vector<sched_t> update_size(prune_policy_t& policy, const sched_t& e0, int layer, int group_num, int source_num, int dest_num) {
  std::vector<sched_t> res;
  std::vector<bool> valid_res;
  assert((source_num+dest_num) % group_num == 0);
  int num_gpu_per_group = (source_num+dest_num)/group_num;
  //assert(policy.prefer_equal_source_per_group && policy.prefer_equal_dest_per_source);

  for (int d_per_s = std::max(1, dest_num/source_num); d_per_s <= std::min(dest_num, layer_num_links_per_gpu(e0.topo, layer)); d_per_s++) {
    // Each comm_source should have same number of destinations
    if (dest_num % d_per_s != 0)
      continue;
    int total_comm_s_num = dest_num / d_per_s;
    // Each group has same number of actual sources
    if (total_comm_s_num % group_num != 0 && !policy.allow_unequal_source_per_group)
      continue;

    // If source_per_group is imbalanced, only allow each comm_source to only one destination
    if (source_num % group_num != 0 && (!policy.allow_unequal_source_per_group || d_per_s != 1))
      continue;
    // If dest_per_source is imbalanced, only allow each source to zero/1/all destinations
    if (d_per_s * source_num != dest_num && (!policy.allow_unequal_dest_per_source || (d_per_s != 1 && d_per_s != dest_num)))
      continue;

    int tmp_comm_source_per_group = total_comm_s_num / group_num;
    int tmp_num_groups = total_comm_s_num % group_num == 0 ? group_num : group_num * (tmp_comm_source_per_group + 1) - total_comm_s_num;
    int acc_num_groups = 0;
    auto e = e0;
    while (acc_num_groups < group_num) {
      int tmp_dest_per_group = tmp_comm_source_per_group * d_per_s;
      int tmp_non_comm_source_per_group = num_gpu_per_group - tmp_comm_source_per_group - tmp_dest_per_group;

      if (tmp_comm_source_per_group != 0) {
        e.per_layer_size[layer].config.emplace_back(tmp_num_groups, tmp_comm_source_per_group,tmp_non_comm_source_per_group, tmp_dest_per_group);
        e.tmp_acc_dest += tmp_dest_per_group * tmp_num_groups;
      }

      tmp_comm_source_per_group += 1;
      acc_num_groups += tmp_num_groups;
      tmp_num_groups = group_num - tmp_num_groups;
    }
    bool e_valid = true;
    for (auto& config : e.per_layer_size.at(layer).config) {
      assert(config.dest_per_group + config.comm_source_per_group + config.non_comm_source_per_group == num_gpu_per_group);
      if(config.non_comm_source_per_group < 0)
        e_valid = false;
    }
    if (e_valid) {
      e.id = ++gid;
      res.push_back(e);
    }
  }
  return res;
}

void update_source_size(prune_policy_t& policy, const sched_t& e0, int layer, int group_num, int source_num, int dest_num, std::vector<sched_t>& res) {
  if (group_num != e0.topo.layer_group_src_dest.at(layer).size() || group_num == 1)
    return;
  // Allow source group be different from other groups
  int num_gpu_per_group = (source_num+dest_num)/group_num;
  int source_group = e0.topo.gpu_layer_group.at(e0.sender).at(layer);
  for (int g0_source_num = 1; g0_source_num < std::min(source_num, num_gpu_per_group); g0_source_num++) {
    int g0_dest_num = num_gpu_per_group - g0_source_num;
    int g0_non_comm_source = num_gpu_per_group - g0_source_num - g0_dest_num;
    if (g0_dest_num % g0_source_num != 0)
      continue;
    int other_group_num = group_num-1;
    if ((source_num-g0_source_num) % other_group_num != 0)
      continue;
    int other_source_num = (source_num-g0_source_num) / other_group_num;
    int other_dest_num = (dest_num-g0_dest_num) / other_group_num;
    int other_non_comm_source = num_gpu_per_group - other_source_num - other_dest_num;
    if (other_source_num == 0 || other_dest_num == 0 || other_dest_num % other_source_num != 0)
      continue;
    if (other_dest_num/other_source_num == g0_dest_num/g0_source_num && other_source_num == g0_source_num)
      continue;
    sched_t e1 = e0;
    e1.id = ++gid;
    e1.per_layer_size[layer].config.emplace_back(1,g0_source_num, g0_non_comm_source, g0_dest_num, source_group);
    e1.per_layer_size[layer].config.emplace_back(other_group_num,other_source_num, other_non_comm_source, other_dest_num, -1);
    e1.tmp_acc_dest += g0_dest_num;
    e1.tmp_acc_dest += other_dest_num * other_group_num;
    res.push_back(e1);
  }
}

// Given used_layers, search for used_groups and group size
void search_sched_groups_func(prune_policy_t& policy, const sched_t& e0, std::vector<sched_t>& outputs) {
  std::queue<sched_t> tmp_q({e0});
  int remained_dest_max, remained_dest_min;
  get_remained_dest(e0, remained_dest_max, remained_dest_min);
  for (auto& [layer, comm] : e0.per_layer_size) {
    int group_num = layer_num_groups(e0.topo, layer);
    int group_gpu_num = layer_num_gpus_per_group(e0.topo, layer);
    update_remained_dest(group_num, group_gpu_num, remained_dest_max, remained_dest_min);
    int n = tmp_q.size();
    for (int j = 0; j < n; j++) {
      auto e = tmp_q.front();
      LOG(ALGO, TRACE, "Search %s", e.toStr().c_str());
      tmp_q.pop();
      int cur_layer_min_dest = std::max(1, e.gpu_num-remained_dest_max-e.tmp_acc_dest);
      int cur_layer_max_dest = std::min(group_num*(group_gpu_num-1), e.gpu_num-remained_dest_min-e.tmp_acc_dest);
      for (int used_group = 1; used_group <= std::min(group_num, cur_layer_max_dest); used_group++) {
        LOG(ALGO, TRACE, "used_group=%d", used_group);
        if (policy.all_groups_used && used_group != group_num)
          continue;
        int dest_min = std::max(cur_layer_min_dest, used_group);
        int dest_max = std::min(cur_layer_max_dest, used_group*(group_gpu_num-1));
        for (int dest_size = dest_min; dest_size <= dest_max; dest_size++) {
          int source_size = group_gpu_num*used_group-dest_size;
          // distribute used_groups and src/dst size to actual groups
          auto e_vec = update_size(policy, e, layer, used_group, source_size, dest_size);
          if (policy.allow_source_group_special) {
            update_source_size(policy, e, layer, used_group, source_size, dest_size, e_vec);
          }
          for (auto& e1 : e_vec) {
            // LOG(ALGO, TRACE, "Explore Sched size %s", e.toStr().c_str());
            if (sched_finish(e1)) {
              if (output_sched_size(policy, e1)) {
                LOG(ALGO, INFO, "Combination generated : %s", e1.toStr().c_str());
                outputs.push_back(e1);
              } else {
                LOG(ALGO, TRACE, "Prune %s", e1.toStr().c_str());
              }
            } else {
              e1.id++;
              tmp_q.push(e1);
              LOG(ALGO, TRACE, "Tmp generated %s", e1.toStr().c_str());
            }
          }
        }
      }
    }
  }
}

std::vector<sched_t> search_sched_groups(prune_policy_t& policy, std::vector<sched_t>& inputs) {
  std::vector<sched_t> outputs;
  LOG(ALGO, INFO, "Search E/C sizes ...");
  for (auto& e : inputs) {
    LOG(ALGO, INFO, "Input Layer combination: %s", e.toStr().c_str());
    search_sched_groups_func(policy, e, outputs);
  }
  for (auto& x : outputs) {
    LOG(ALGO, TRACE, "Generate: %s", x.toStr().c_str());
  }
  LOG(ALGO, INFO, "%d generated\n", outputs.size());
  return outputs;
}
