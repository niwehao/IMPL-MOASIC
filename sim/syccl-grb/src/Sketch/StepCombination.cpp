// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/searchutils.hpp"
#include <algorithm>
#include <set>
#include "common/ctpl.h"

// Step 3: for each group configuration, determine the communication steps
void update_scheduling(sched_t& e, const std::set<int>& sources, const std::set<int>& dests,
                       int step, int layer, int group, const layer_prune_t& p) {
  for (int dest : dests) {
    e.comm_by_gpus[dest] = {step, layer, group, sources};
    e.nxt_step_sources.insert(dest);
    // auto dev = topo->npus[dest];
    // for (auto g : topo->topo_info.device_groups[dev.device]) {
    //   e.comm_by_layer[g.first].group_finished_gpus[g.second]++;
    // }
  }
  e.layer_prunes[layer] = p;
  if (!group_has_comm(e, layer, group)) {
    int config_id = new_group_config_id(e, layer, group);
    assert(config_id != -1);
    e.comm_by_layer[layer].config_id_to_groups[config_id].insert(group);
    e.comm_by_layer[layer].group_to_config_id[group] = config_id;
  }
  e.comm_by_layer[layer].comm_groups[group].src_dest_pairs.push_back({sources, dests});
  e.comm_by_step[step].layer_group_comm[layer][group].src_dest_pairs.push_back({sources, dests});
}

void search_destinations(const sched_t& e, int layer, int group, std::set<int>& source_comb,
                         std::set<int>& dsts, std::map<int, std::set<int>>& src_to_dst) {
  for (auto src : source_comb) {
    for (auto dst: e.topo.layer_group_src_dest.at(layer).at(group).at(src)) {
      if (!gpu_has_comm(e, dst)) {
        dsts.insert(dst);
        src_to_dst[src].insert(dst);
      }
    }
  }
}

sched_t copy_scheduling(const sched_t& e0, const std::set<int>& sources, const std::set<int>& dests,
                              int step, int layer, int group, const layer_prune_t& p) {
  // sched_t e = copy_scheduling(e0);
  auto e = e0;
  update_scheduling(e, sources, dests, step, layer, group, p);
  return e;
}

sched_t copy_scheduling(const sched_t& e0, const std::set<int>& sources, const std::set<int>& dests,
                        int step, int layer, int group) {
  // sched_t e = copy_scheduling(e0);
  auto p = get_layer_prune(e0, layer);
  return copy_scheduling(e0, sources, dests, step, layer, group, p);
}

mapKey get_gpu_key(const sched_t& e, int gpu) {
  mapKey res;
  if (gpu_has_comm(e, gpu))
    res.comm_paths = comm_path_layers(e, gpu);
  for (auto& [layer, conf] : e.per_layer_size) {
    int group = gpu_group_in_layer(e.topo, gpu, layer);
    res.layer_gpus[layer] = group_num_reached_gpus(e, layer, group);
  }
  return res;
}

std::map<mapKey, std::set<int>> partition_gpus(const sched_t& e, const std::set<int>& gpus) {
  std::map<mapKey, std::set<int>> res;
  for (auto& gpu : gpus) {
    res[get_gpu_key(e, gpu)].insert(gpu);
  }
  return res;
}

std::string to_str(const std::set<int>& s) {
  std::string res;
  for (auto a : s)
    res += std::to_string(a) + ",";
  res.pop_back();
  return res;
}

struct dest_comb_prune_t {
  std::map<int, std::multiset<int>> layer_group_finished_gpu_num;
  double balance_ratio = -1;
  bool is_larger_than(dest_comb_prune_t& e1) {
    for (auto& it : layer_group_finished_gpu_num) {
      int layer = it.first;
      auto it1 = it.second.rbegin();
      auto it2 = e1.layer_group_finished_gpu_num[layer].rbegin();
      while (it1 != it.second.rend() && it2 != e1.layer_group_finished_gpu_num[layer].rend()) {
        if (*it1 > *it2)
          return true;
        else if (*it1 < *it2)
          return false;
        it1++; it2++;
      }
    }
    return false;
  }
};

double get_mean_dev(std::multiset<int>& data) {
  assert(!data.empty());
  int sum_val = 0;
  for (auto d : data)
    sum_val += d;
  double avg = (float)sum_val / data.size();
  double total_dev = 0;
  for (auto d : data)
    total_dev += (d-avg) * (d-avg);
  return total_dev/data.size();
}

dest_comb_prune_t get_ratio(const sched_t& e0, int step, int layer, int group,
                            const std::set<int>& source_comb, const std::set<int>& dest_comb) {
  auto e = copy_scheduling(e0, source_comb, dest_comb, step, layer, group);
  dest_comb_prune_t res;
  for (auto& [other_layer, config] : e.per_layer_size) {
    if (layer == other_layer) continue;
    for (auto& [group, src_dest] : e0.topo.layer_group_src_dest.at(other_layer)) {
      int finished_gpu_num = group_num_reached_gpus(e, other_layer, group);
      res.layer_group_finished_gpu_num[other_layer].insert(finished_gpu_num);
    }
    res.balance_ratio += get_mean_dev(res.layer_group_finished_gpu_num[other_layer]);
  }
  return res;
}

std::map<dst_prune_e, vec_set_t> merge_prunes(std::map<dst_prune_e, std::set<int>>& indexes, const vec_set_t& dest_combs) {
  std::map<dst_prune_e, vec_set_t> res;
  for (auto& i : indexes[dst_prune_balance]) {
    if (indexes[dst_prune_imbalance].find(i) != indexes[dst_prune_imbalance].end()) {
      res[dst_prune_undefine].push_back(dest_combs[i]);
      indexes[dst_prune_imbalance].erase(i);
    } else {
      res[dst_prune_balance].push_back(dest_combs[i]);
    }
  }
  for (auto& i : indexes[dst_prune_imbalance]) {
    res[dst_prune_imbalance].push_back(dest_combs[i]);
  }
  return res;
}

std::map<dst_prune_e, vec_set_t> prune_dest_combs(prune_policy_t& policy, const sched_t& e0, int step, int layer, int group,
                           const std::set<int>& source_comb, const vec_set_t& dest_combs) {
  if(!policy.prune_dests) {
    std::map<dst_prune_e, vec_set_t> res;
    res[dst_prune_undefine] = dest_combs;
    return res;
  }
  dest_comb_prune_t res_ratio;
  std::map<dst_prune_e, std::set<int>> res_index;
  for (int i = 0; i < dest_combs.size(); i++) {
    dest_comb_prune_t attr = get_ratio(e0, step, layer, group, source_comb, dest_combs[i]);
    auto dst_prune = get_layer_prune(e0, layer).dst_prune;

    // all groups are most balanced
    if (dst_prune == dst_prune_undefine || (policy.allow_dest_balanced && dst_prune == dst_prune_balance)) {
      if (res_ratio.balance_ratio < 0 || attr.balance_ratio < res_ratio.balance_ratio) {
        res_ratio.balance_ratio = attr.balance_ratio;
        res_index[dst_prune_balance].clear();
        res_index[dst_prune_balance].insert(i);
      } else if (attr.balance_ratio == res_ratio.balance_ratio) {
        res_index[dst_prune_balance].insert(i);
      }
    }

    // centralized in some groups
    if (dst_prune == dst_prune_undefine || (policy.allow_dest_centralized && dst_prune == dst_prune_imbalance)) {
      if (res_ratio.layer_group_finished_gpu_num.empty() || attr.is_larger_than(res_ratio)) {
        res_ratio.layer_group_finished_gpu_num = attr.layer_group_finished_gpu_num;
        res_index[dst_prune_imbalance].clear();
        res_index[dst_prune_imbalance].insert(i);
      } else if (attr.is_larger_than(res_ratio)) {
        res_index[dst_prune_imbalance].insert(i);
      }
    }
  }
  return merge_prunes(res_index, dest_combs);
}

void search_dests(prune_policy_t& policy, const sched_t& e, std::set<int>& source_comb,
                  int step, int layer, int group, int source_per_group, const layer_prune_t& prune_info,
                  std::queue<sched_t>& res) {
  std::set<int> dsts;
  std::map<int, std::set<int>> src_to_dsts;
  search_destinations(e, layer, group, source_comb, dsts, src_to_dsts);
  vec_set_t dest_combs;
  if (policy.partition_dests) {
    auto dest_partitions = partition_gpus(e, dsts);
    int dest_num = layer_d_per_s(e, layer, group);
    dest_num *= source_comb.size();
    dest_combs = gpu_combinations(dest_num, dsts, dest_partitions);
    if (dest_combs.size() > 3)
      LOOOG(ALGO, TRACE, "Note!!!");
  } else {
    dest_combs.push_back(dsts);
  }
  auto pruned_dest_combs = prune_dest_combs(policy, e, step, layer, group, source_comb, dest_combs);
  for (auto& dst_prune_res : pruned_dest_combs) {
    // LOG(ALGO, TRACE, "\t\t\t\tsearch_dests: Group %d, %d dest combinations => prune into %d combs", group, dest_combs.size(), pruned_dest_combs.size());
    for (auto &dest_comb: dst_prune_res.second) {
      auto new_sched = copy_scheduling(e, source_comb, dest_comb, step, layer, group, prune_info);
      assert(new_sched.layer_prunes[layer].dst_prune == dst_prune_undefine || new_sched.layer_prunes[layer].dst_prune == dst_prune_res.first);
      new_sched.layer_prunes[layer].dst_prune = dst_prune_res.first;
      res.push(new_sched);
      LOG(ALGO, TRACE, "\t\t\tsearch_dests: {%s} -> {%s}", to_str(source_comb).c_str(), to_str(dest_comb).c_str());
    }
  }
}

// Divide groups by allowed num_sources
map_set_t partition_groups(const sched_t& e, int layer, std::map<int, std::set<int>>& sources, std::map<int, std::set<int>>& dests) {
  map_set_t res;
  for (auto& [group, src] : sources) {
    int max_source_num = src.size();
    int group_cur_dest_gpus = group_num_dest(e, layer, group);
    int group_obj_dest = group_num_obj_dest(e, layer, group);
    int max_allow_num = std::min(max_source_num, (group_obj_dest-group_cur_dest_gpus) / layer_d_per_s(e, layer, group));
    res[max_allow_num].insert(group);
  }
  return res;
}

// Search num_groups and num_sources_per_group used in current step and layer
struct layer_size_t {
    int layer;
    int source_per_group;
    int group_num;
};

std::vector<layer_size_t> search_layer_sizes(const sched_t& e, int layer, const map_set_t& group_partitions) {
  std::vector<layer_size_t> res;
  layer_size_t source_comb = {layer};

  int max_source_per_group = 0;
  for (auto& conf : e.per_layer_size.at(layer).config) {
    if (conf.comm_source_per_group > max_source_per_group)
      max_source_per_group = conf.comm_source_per_group;
  }
  if (max_source_per_group > group_partitions.rbegin()->first)
    max_source_per_group = group_partitions.rbegin()->first;

  int acc_group_num = 0;
  for (auto it: group_partitions)
    acc_group_num += it.second.size();

  auto group_it = group_partitions.begin();
  for (int source_per_group = 1; source_per_group <= max_source_per_group; source_per_group++) {
    source_comb.source_per_group = source_per_group;

    if (source_per_group > group_it->first) {
      acc_group_num -= group_it->second.size();
      group_it++;
      if (group_it == group_partitions.end())
        break;
    }
    int max_group_num = std::min(acc_group_num, total_num_groups(e, layer));
    for (int group_num = 1; group_num <= max_group_num; group_num++) {
      source_comb.group_num = group_num;
      res.push_back(source_comb);
    }
  }
  return res;
}

// Prune {num_groups and num_sources_per_group} by cross-step numbers
void update_src_prune(layer_prune_t& p, int step, int group_num, int source_per_group) {
  if (p.num_group_per_step.size() < step) {
    if (p.step_split != -1)
      assert(p.num_group_per_step.size() + p.step_split == step+1);
    if (!p.num_group_per_step.empty())
      p.step_split = step+1 - p.num_group_per_step.size();
    for (int s = p.num_group_per_step.size(); s < step; s++) {
      p.num_group_per_step.push_back(0);
      p.num_source_per_group_per_step.push_back(0);
    }
  }
  p.num_group_per_step.push_back(group_num);
  p.num_source_per_group_per_step.push_back(source_per_group);
}

void check_arithmetric_or_geometric(std::vector<int>& numbers, bool& is_arithmetric, bool& is_geometric, int step_split) {
  if (step_split == -1)
    step_split = 1;
  for (int s = 0; s < numbers.size(); s++) {
    if (is_arithmetric && s >= 2*step_split) {
      if ((numbers[s]+numbers[s-2*step_split]) != (numbers[s-step_split]+numbers[s-step_split]))
        is_arithmetric = false;
    }
    if (is_geometric && s >= 2*step_split) {
      if ((numbers[s]*numbers[s-2*step_split]) != (numbers[s-step_split]*numbers[s-step_split]))
        is_geometric = false;
    }
  }
}

bool prune_source_arithmetric_or_geometric(prune_policy_t& policy, layer_prune_t& p) {
  if (!policy.prune_sources_by_compare_steps)
    return false;
  bool is_arithemetric = policy.allow_source_per_group_arithmetic_per_step;
  bool is_geometric = policy.allow_source_per_group_geometric_per_step;
  check_arithmetric_or_geometric(p.num_source_per_group_per_step, is_arithemetric, is_geometric, p.step_split);
  if (!is_arithemetric && !is_geometric)
    return true;
  is_arithemetric = policy.allow_num_groups_arithmetic_per_step;
  is_geometric = policy.allow_num_groups_geometric_per_step;
  check_arithmetric_or_geometric(p.num_group_per_step, is_arithemetric, is_geometric, p.step_split);
  if (!is_arithemetric || !is_geometric)
    return true;
  return false;
}

std::vector<std::pair<layer_prune_t, layer_size_t>> prune_layer_sizes(prune_policy_t& policy, const sched_t& e,
                                                                    const std::vector<layer_size_t>& orig_sizes, int step) {
  // Each layer should have the group_num and source_per_group in an arithmetic or geometric progression at different steps
  std::vector<std::pair<layer_prune_t, layer_size_t>> res;
  int max_num_group = -1, max_num_source = -1;
  for (auto& s : orig_sizes) {
    max_num_group = std::max(max_num_group, s.group_num);
    max_num_source = std::max(max_num_source, s.source_per_group);
  }
  for (auto& s : orig_sizes) {
    int layer = s.layer;
    auto prune_info = get_layer_prune(e, layer);
    update_src_prune(prune_info, step, s.group_num, s.source_per_group);
    if (prune_source_arithmetric_or_geometric(policy, prune_info))
      continue;
    if (policy.drop_small_sources && max_num_source != s.source_per_group &&
      (!policy.keep_large_groups || max_num_group != s.group_num))
      continue;
    if (policy.drop_small_groups && max_num_group != s.group_num &&
      (!policy.keep_large_sources || max_num_source != s.source_per_group))
      continue;
    res.push_back(std::make_pair(prune_info, s));
  }
  return res;
}

// Search specific groups
vec_set_t search_layer_groups(const map_set_t& group_partitions, const layer_size_t& layer_size) {
  std::set<int> g0;
  std::queue<std::set<int>> tmp_q({g0});

  int total_group_num = 0;
  for (auto& it : group_partitions)
    total_group_num += it.second.size();

  for (auto& it : group_partitions) {
    total_group_num -= it.second.size();
    if (it.first < layer_size.source_per_group)
      continue;

    int n = tmp_q.size();
    for (int i = 0; i < n; i++) {
      auto groups = tmp_q.front();
      tmp_q.pop();
      int group_num_to_add = layer_size.group_num - groups.size();
      int min_group_num = group_num_to_add > total_group_num ? group_num_to_add-total_group_num : 0;
      int max_group_num = group_num_to_add > total_group_num ? std::min(group_num_to_add, int(it.second.size())) : 0;
      for (int j = min_group_num; j <= max_group_num; j++) {
        // LOG(ALGO, TRACE, "\t\t\t\tsearch_layer_groups: %d groups out of {%s}", j, to_str(it.second).c_str());
        auto groups_new = groups;
        int k = 0;
        for (auto group : it.second) {
          groups_new.insert(group);
          k++;
          if (k == j) break;
        }
        tmp_q.push(groups_new);
      }
    }
  }
  vec_set_t res;
  while(!tmp_q.empty()) {
    res.push_back(tmp_q.front());
    tmp_q.pop();
  }
  return res;
}

// Search sources and destinations inside groups
vec_set_t search_sources(prune_policy_t& policy, const sched_t& e, const map_set_t& group_sources, int group, int source_per_group) {
  std::set<int> sources = group_sources.at(group);
  vec_set_t res;
  if (policy.partition_sources) {
    auto source_partitions = partition_gpus(e, sources);
    res = gpu_combinations(source_per_group, sources, source_partitions);
  } else {
    res.push_back(sources);
  }
  // LOG(ALGO, TRACE, "\t\t\t\tsearch_sources: Group %d, %d source combinations", group, res.size());
  return res;
}

bool prune_sources(prune_policy_t& policy, const sched_t& e0,
                   int step, int layer, int group, std::set<int>& sources) {
  if (!step_has_comm(e0, step) || e0.comm_by_step.at(step).layer_group_comm.find(layer) == e0.comm_by_step.at(step).layer_group_comm.end())
    return false;
  if (gpu_group_in_layer(e0.topo, e0.sender, layer) == group)
    return false;
  int g1 = e0.topo.layer_group_src_dest.at(layer).at(group).begin()->first;
  int g2 = e0.topo.layer_group_src_dest.at(layer).at(group).rbegin()->first;
  if (g1 == g2)
    return false;
  for (auto& [group_id, comm] : e0.comm_by_step.at(step).layer_group_comm.at(layer)) {
    if (group_id < group && gpu_group_in_layer(e0.topo, e0.sender, layer) != group_id) {
      int other_g1 = e0.topo.layer_group_src_dest.at(layer).at(group_id).begin()->first;
      int other_g2 = e0.topo.layer_group_src_dest.at(layer).at(group_id).rbegin()->first;
      int source_num = comm.src_dest_pairs.size();
      if (sources.size() > comm.src_dest_pairs.size())
        return false;
      if (source_num == sources.size()) {
        auto it1 = sources.begin();
        auto it2 = comm.src_dest_pairs.begin();
        while (it1 != sources.end()) {
          int i1 = (*it1 - g1) / (g2 - g1);
          int i2 = (*(it2->first.begin()) - other_g1) / (other_g2 - other_g1);
          if (i1 > i2)
            return true;
          it1++;
          it2++;
        }
      }
    }
  }
  return false;
}

void search_layer_func(prune_policy_t& policy, const sched_t& e0, int layer, const std::set<int>& groups, const map_set_t& group_sources,
                       int step, const layer_prune_t& prune_info, int source_per_group, std::queue<sched_t>& res) {
  /*
  Given group index, source_per_group and the source GPUs in this group, search the source GPU combinations and store them in source_combs
  Prune the source GPU combinations in source_combs
  Search the destination GPUs to generate the schedule in this (step, layer, group)
  */
  std::queue<sched_t> tmp_q({e0});
  for (int group : groups) {
    int n = tmp_q.size();
    for (int i = 0; i < n; i++) {
      auto e = tmp_q.front();
      tmp_q.pop();
      auto source_combs = search_sources(policy, e, group_sources, group, source_per_group);
      for (auto &source_comb: source_combs) {
        if (prune_sources(policy, e, step, layer, group, source_comb))
          continue;
        search_dests(policy, e, source_comb, step, layer, group, source_per_group, prune_info, tmp_q);
        if (policy.max_dest_combs != -1 && n > policy.max_dest_combs)
          break;
      }
    }
  }
  while(!tmp_q.empty()) {
    tmp_q.front().id = ++gid;
    LOG(ALGO, TRACE, "\t\t\t\tsearch_layer_func: new sched [#%d]", gid);
    res.push(tmp_q.front());
    tmp_q.pop();
  }
}

bool is_valid_groups(const sched_t& e, int layer, std::set<int>& groups) {
  int group_num = 0;
  if (!layer_has_comm(e, layer))
    group_num = groups.size() ;
  else {
    group_num = e.comm_by_layer.at(layer).comm_groups.size();
    for (auto group: groups) {
      if (!group_has_comm(e, layer, group)) {
        group_num++;
      }
    }
  }
  return group_num <= total_num_groups(e, layer);
}

void search_layer(prune_policy_t& policy, int layer, map_set_t& sources, map_set_t& dests,
                  const sched_t& e, int step, std::queue<sched_t>& res) {
  /*
  In this layer, search how many groups can be used (i.e. group_num) and how many sources per group can be used (i.e. source_per_group)
  The total number of combinations is group_num * source_per_group, stored in layer_sizes
  Then prune some combinations in layer_sizes to reduce the searching space and finally generate pruned_layer_sizes
  */
  //assert(policy.partition_source_groups);
  auto group_partitions = partition_groups(e, layer, sources, dests);
  LOG(ALGO, TRACE, "\t\t\t\tpartition_groups: %d source groups => %d partitions", sources.size(), group_partitions.size());

  // Search group combinations
  auto layer_sizes = search_layer_sizes(e, layer, group_partitions);
  LOG(ALGO, TRACE, "\t\t\t\tsearch_layer_sizes: %d group_partitions => %d layer_sizes", group_partitions.size(), layer_sizes.size());

  auto pruned_layer_sizes = prune_layer_sizes(policy, e, layer_sizes, step);
  LOG(ALGO, TRACE, "\t\t\t\tsearch_layer: Prune layer sizes from %d to %d", layer_sizes.size(), pruned_layer_sizes.size());

  for (auto& it : pruned_layer_sizes) {
    auto src_prune = it.first;
    auto layer_size = it.second;
    auto layer_groups = search_layer_groups(group_partitions, layer_size);
    LOG(ALGO, TRACE, "\t\t\tsearch_layer: layer_size: %d groups * %d sources, generate %d layer groups",
        layer_size.group_num, layer_size.source_per_group, layer_groups.size());
    for (auto &groups: layer_groups) {
      if (is_valid_groups(e, layer, groups))
        search_layer_func(policy, e, layer, groups, sources, step, src_prune, layer_size.source_per_group, res);
    }
  }
}

bool search_start_sources(prune_policy_t& policy, const sched_t& e, int step, std::vector<layer_cand_t>& start_sources) {
  /* 
  Generate sources and destinations in different layers and groups, e.g.
  L#1, Group#0, {0} -> {1,2,3,4,5,6,7}
  L#3, Group#0, {0} -> {8,9,10,11,12,13,14,15}
  L#4, Group#0, {0} -> {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
  */
  for (auto &[layer, conf]: e.per_layer_size) {
    // Larger layers cannot already comm in same step
    bool layer_already_searched = false;
    if (step_has_comm(e, step)) {
      for (auto &[other_layer, comm]: e.comm_by_step.at(step).layer_group_comm)
        if (other_layer >= layer) {
          layer_already_searched = true;
          break;
        }
    }
    if (layer_already_searched) continue;
    // Each layer should be used in certain steps
    if (policy.same_step_diff) {
      if (e.layer_prunes.find(layer) != e.layer_prunes.end()) {
        if (e.layer_prunes.at(layer).step_split != -1 &&
          e.layer_prunes.at(layer).num_group_per_step.size() + e.layer_prunes.at(layer).step_split != step + 1) {
          continue;
        }
      }
    }
    layer_cand_t s;
    s.layer = layer;
    // assert(policy.only_next_step_source);
    for (int src: e.cur_step_sources) {
      int group = gpu_group_in_layer(e.topo, src, layer);
      if (policy.start_source_diff_layer && e.comm_by_gpus.at(src).layer == layer)
        continue;
      // Remained GPUs as destinations are enough
      if (group_has_comm(e, layer, group)) {
        int group_not_reached_gpus = group_num_not_reached_gpus(e, layer, group);
        int group_cur_dest_gpus = group_num_dest(e, layer, group);
        int group_conf_id = e.comm_by_layer.at(layer).group_to_config_id.at(group);
        int group_obj_dest = e.per_layer_size.at(layer).config.at(group_conf_id).dest_per_group;
        if (group_not_reached_gpus < (group_obj_dest - group_cur_dest_gpus)) {
          return false; // Prune
        } else if (group_obj_dest == group_cur_dest_gpus) {
          continue;
        }
      } else {
        if (layer_has_comm(e, layer) &&
            e.comm_by_layer.at(layer).comm_groups.size() == total_num_groups(e, layer))
          continue;
      }
      // add all destination GPUs that have not been used
      for (auto& gpu: e.topo.layer_group_src_dest.at(layer).at(group).at(src)) {
        if (!gpu_has_comm(e, gpu))
          s.group_dests[group].insert(gpu);
      }
      
      if (s.group_dests.find(group) != s.group_dests.end()) {
        s.group_sources[group].insert(src);
      }
    }
    if (!s.group_sources.empty())
      start_sources.push_back(s);
  }
  return true;
}

bool output_sched(const sched_t& e) {
  return e.comm_by_gpus.size() == e.gpu_num;
}

void search_step(prune_policy_t& policy, const sched_t& e0, std::queue<sched_t>& next_step_queue, int step) {
  std::queue<sched_t> tmp_q({e0});
  while (!tmp_q.empty()) {
    auto e = tmp_q.front();
    tmp_q.pop();

    // Determine all sources and destinations (and layers) that can be used to comm in current step
    std::vector<layer_cand_t> start_src_dests;
    if (search_start_sources(policy, e, step, start_src_dests)) {
      for (auto &src_dests: start_src_dests) {
        for (auto &[group, sources]: src_dests.group_sources) {
          LOG(ALGO, TRACE, "\t\tL#%d, Group#%d, {%s -> %s}", src_dests.layer, group,
              to_str(src_dests.group_sources[group]).c_str(), to_str(src_dests.group_dests[group]).c_str());
        }
        if (src_dests.group_sources.empty())
          continue;
        search_layer(policy, src_dests.layer, src_dests.group_sources, src_dests.group_dests, e, step, tmp_q);
      }
    }

    // search next step
    if (step_has_comm(e, step)) {
      e.cur_step_sources = e.nxt_step_sources;
      e.nxt_step_sources.clear();
      if (!e.cur_step_sources.empty() || output_sched(e)) {
        LOG(ALGO, TRACE, "\t\tsearch_one_step: [#%d] to next step !!!!", e.id);
        next_step_queue.push(e);
      }
    }
  }
}

std::mutex output_mutex;
void search_sched_steps_func(prune_policy_t& policy, const sched_t& e0, std::vector<sched_t>& outputs) {
  LOG(ALGO, TRACE, "\n\n\tsearch_sched_steps_func: %s", e0.toStr().c_str());
  std::queue<sched_t> tmp_q({e0});
  int total_group = total_num_groups(e0);
  int max_step = policy.limit_num_steps != -1 ? policy.limit_num_steps :
          policy.limit_num_steps_ratio != -1 ? total_group*policy.limit_num_steps_ratio : e0.gpu_num - 1;
  for (int step = 0; step <= max_step; step++) {
    if (tmp_q.empty())
      break;
    int size_to_search = tmp_q.size();
    for (int i = 0; i < size_to_search; i++) {
      auto e = tmp_q.front();
      tmp_q.pop();
      // If current schedule e complete all communications
      if (output_sched(e)) {
        LOG(ALGO, TRACE, "\n\tsearch_func: step %d, %d/%d", step, i+1, size_to_search);
        LOG(ALGO, TRACE, "Generated Output %s", e.toStr().c_str());
        std::lock_guard<std::mutex> lock(output_mutex);
        outputs.push_back(e);
      } else if (step < max_step) {
        LOG(ALGO, TRACE, "\n\tsearch_sched_steps_func: step %d, search %d/%d", step, i+1, size_to_search);
        LOG(ALGO, TRACE, "output %s", e.toStr().c_str());
        search_step(policy, e, tmp_q, step);
      }
    }
  }
}

#define SEARCH_STEP_MULTI_THREAD_ENABLE
// #undef SEARCH_STEP_MULTI_THREAD_ENABLE
std::vector<sched_t> search_sched_steps(prune_policy_t& policy, std::vector<sched_t>& inputs) {
  std::vector<sched_t> outputs;
  LOOOG(ALGO, INFO, "Search Steps ...");
#ifdef SEARCH_STEP_MULTI_THREAD_ENABLE
  std::vector<sched_t> tmp_res;
  ctpl::thread_pool p(SKETCH_EXPLORATION_THREAD_NUM);
#endif
  int i = 1;
  for (auto& e : inputs) {
    gid = 0;
#ifdef SEARCH_STEP_MULTI_THREAD_ENABLE
    LOG(ALGO, TRACE, "Search %d/%d", i++, inputs.size());
    LOG(ALGO, TRACE, "%s", e.toStr().c_str());
    p.push([&](int id) {
      search_sched_steps_func(std::ref(policy), std::ref(e), std::ref(tmp_res));
    });
#else
    int before_size = outputs.size();
    LOG(ALGO, INFO, "Searching Steps For Input %s", e.toStr().c_str());
    search_sched_steps_func(policy, e, outputs);
    // this log is not for multi threads
    LOG(ALGO, INFO, "%d generated, %d total", outputs.size() - before_size, outputs.size());
    for (int i = before_size; i < outputs.size(); i++) {
      LOG(ALGO, INFO, "Output %d/%d %s", i+1, outputs.size(), outputs[i].toStr().c_str());
    }
#endif
  }
#ifdef SEARCH_STEP_MULTI_THREAD_ENABLE
  p.stop(true);
  for (auto& e : tmp_res) {
    outputs.push_back(e);
  }
#endif

  i = 1;
  for (auto& e : outputs) {
    LOG(ALGO, TRACE, "Output %d/%d %s", i++, outputs.size(), e.toStr().c_str());
  }
  LOG(ALGO, INFO, "search_sched_steps: %d generated\n", outputs.size());
  return outputs;
}
