// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sketch/Algorithm.h"
#include <algorithm>
#include <set>

typedef std::shared_ptr<Simulation::Topology> topo_t;
typedef std::map<int, std::set<int>> map_set_t;
typedef std::vector<std::set<int>> vec_set_t;

struct mapKey {
    std::vector<std::vector<int>> comm_paths;
    std::map<int, int> layer_gpus;
    bool operator<(const mapKey& other) const {
      if (this->comm_paths == other.comm_paths) {
        return this->layer_gpus < other.layer_gpus;
      }
      return this->comm_paths < other.comm_paths;
    }
};

template <typename T>
inline vec_set_t gpu_combinations(int num, std::set<int>& gpus, const std::map<T, std::set<int>>& partitions) {
  std::map<int, T> p1;
  for (auto& it : partitions)
    for (auto gpu : it.second)
      p1[gpu] = it.first;

  vec_set_t res;
  std::set<int> tmp;
  std::queue<std::set<int>> tmp_q({tmp});
  int remained_gpu_num = gpus.size();
  for (int gpu : gpus) {
    int n = tmp_q.size();
    remained_gpu_num--;
    for (int i = 0; i < n; i++) {
      auto e = tmp_q.front();
      tmp_q.pop();

      // gpu not chosen
      if (remained_gpu_num + e.size() >= num) {
        tmp_q.push(e);
      }
      // gpu chosen
      bool is_valid = true;
      for (auto gpu1 : partitions.at(p1[gpu])) {
        if (gpu1 < gpu && e.find(gpu1) == e.end()) {
          is_valid = false;
          break;
        }
      }
      if (is_valid) {
        e.insert(gpu);
        if (e.size() == num)
          res.push_back(e);
        else
          tmp_q.push(e);
      }
    }
  }
  // assert(tmp_q.empty());
  return res;
}

// Step, layer, group have valid communication
inline bool step_has_comm(const sched_t& e, int step) {
  return e.comm_by_step.find(step) != e.comm_by_step.end();
}

inline bool layer_has_comm(const sched_t& e, int layer) {
  return e.comm_by_layer.find(layer) != e.comm_by_layer.end();
}
inline bool group_has_comm(const sched_t& e, int layer, int group) {
  return layer_has_comm(e, layer) &&
    e.comm_by_layer.at(layer).comm_groups.find(group) != e.comm_by_layer.at(layer).comm_groups.end();
}

inline bool gpu_has_comm(const sched_t& e, int gpu) {
  return e.comm_by_gpus.find(gpu) != e.comm_by_gpus.end();
}

inline int group_num_dest(const sched_t& e, int layer, int group) {
  if (!group_has_comm(e, layer, group))
    return 0;
  int n = 0;
  for (auto &[src, dst]: e.comm_by_layer.at(layer).comm_groups.at(group).src_dest_pairs)
      n += dst.size();
  return n;
}

inline int new_group_config_id(const sched_t& e, int layer, int group) {
  if (!layer_has_comm(e, layer))
    return 0;
  for (int config_id = 0; config_id < e.per_layer_size.at(layer).config.size(); config_id++) {
    if (e.per_layer_size.at(layer).config.at(config_id).group_id != -1 && e.per_layer_size.at(layer).config.at(config_id).group_id != group)
      continue;
    if (e.comm_by_layer.at(layer).config_id_to_groups.find(config_id) ==
        e.comm_by_layer.at(layer).config_id_to_groups.end() ||
        e.comm_by_layer.at(layer).config_id_to_groups.at(config_id).size() <
        e.per_layer_size.at(layer).config.at(config_id).used_group_num)
      return config_id;
  }
  assert(0);
  return -1;
}

inline int group_num_obj_dest(const sched_t& e, int layer, int group) {
  if (!layer_has_comm(e, layer) || !group_has_comm(e, layer, group)) {
    int config_id = new_group_config_id(e, layer, group);
    assert(config_id != -1);
    return e.per_layer_size.at(layer).config.at(config_id).dest_per_group;
  } else {
    int config_id = e.comm_by_layer.at(layer).group_to_config_id.at(group);
    return e.per_layer_size.at(layer).config.at(config_id).dest_per_group;
  }
}

inline int group_num_not_reached_gpus(const sched_t& e, int layer, int group) {
  int n = 0;
  for (auto& [gpu, dst] : e.topo.layer_group_src_dest.at(layer).at(group)) {
    if (!gpu_has_comm(e, gpu))
      n++;
  }
  return n;
}

inline int total_num_groups(const sched_t& e, int layer) {
  if (e.per_layer_size.find(layer) == e.per_layer_size.end())
    return 0;
  int num_groups = 0;
  for (auto& group_config : e.per_layer_size.at(layer).config)
    num_groups += group_config.used_group_num;
  return num_groups;
}

inline int total_num_groups(const sched_t& e) {
  int num_groups = 0;
  for (auto& [layer, config ] : e.per_layer_size) {
    num_groups += total_num_groups(e, layer);
  }
  return num_groups;
}

// except sender
inline int group_num_reached_gpus(const sched_t& e, int layer, int group) {
  int n = 0;
  for (auto& [gpu, dst] : e.topo.layer_group_src_dest.at(layer).at(group)) {
    if (gpu_has_comm(e, gpu))
      n++;
  }
  return n;
}

inline std::vector<std::vector<int>> comm_path_layers(const sched_t& e, int gpu) {
  std::vector<std::vector<int>> res;
  res.emplace_back();
  if (!gpu_has_comm(e, gpu) || e.comm_by_gpus.at(gpu).step == -1)
    return res;
  std::vector<sched_source_t> individual_sources;
  for (auto src : e.comm_by_gpus.at(gpu).sources) {
    bool same_source = false;
    for (auto p: individual_sources) {
      if (e.comm_by_gpus.at(src).equal(p)) {
        same_source = true;
        break;
      }
    }
    if (!same_source) {
      individual_sources.emplace_back(e.comm_by_gpus.at(src));
      auto paths = comm_path_layers(e, src);
      for (auto p : paths) {
        p.push_back(e.comm_by_gpus.at(gpu).layer);
        res.push_back(p);
      }
    }
  }
  return res;
}

inline sched_t copy_scheduling(const sched_t& e) {
  sched_t e1 = e;
  e1.id = ++gid;
  return e1;
}

inline layer_prune_t get_layer_prune(const sched_t& e, int layer) {
  layer_prune_t p0;
  if (e.layer_prunes.find(layer) != e.layer_prunes.end())
    return e.layer_prunes.at(layer);
  return p0;
}

// modify operations
inline int max_dest_per_source(int source, int dest) {
  int d_per_source = -1;
  int max_d_per_s = dest;
  int min_d_per_s = (dest+source-1)/source;
  for (int i = max_d_per_s; i >= min_d_per_s; i--) {
    if (dest % i == 0)
      d_per_source = i;
  }
  assert(d_per_source != -1);
  return d_per_source;
}

inline int layer_d_per_s(const sched_t& e, int layer, int group) {
  int config_id;
  if (group_has_comm(e, layer, group))
    config_id = e.comm_by_layer.at(layer).group_to_config_id.at(group);
  else {
    config_id = new_group_config_id(e, layer, group);
  }
  return e.per_layer_size.at(layer).config.at(config_id).dest_per_group /
    e.per_layer_size.at(layer).config.at(config_id).comm_source_per_group;
}

inline int gpu_group_in_layer(const topo_subset_t& topo, int gpu_id, int layer) {
  return topo.gpu_layer_group.at(gpu_id).at(layer);
}