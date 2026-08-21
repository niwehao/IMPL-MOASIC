// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "PerfModel/Topology.h"

using namespace Simulation;

void check_group_validity(network_group_t& group) {
  assert(!group.devices.empty());
  for (const auto& node : group.devices) {
    assert((*group.devices.begin())->type == node->type);
  }
  if (group.switch_device == nullptr)
    assert((*group.devices.begin())->type == NODE_GPU);
}

std::string convert_path_to_str(Topology* topo, std::vector<std::shared_ptr<Link>>& links) {
  std::string str;
  str += topo->devices[links[0]->getSrc()]->toStr();
  for (auto link : links) {
    str += "," + topo->devices[link->getDst()]->toStr();
  }
  return str;
}

std::vector<std::shared_ptr<Link>> merge_paths(const std::vector<std::vector<std::shared_ptr<Link>>>& paths) {
  std::vector<std::shared_ptr<Link>> res;
  for (auto& x: paths) {
    assert(!x.empty());
    for (const auto& y: x)
      res.push_back(y);
  }
  return res;
}

bool is_path_overlap(std::vector<std::shared_ptr<Link>>& path1, std::vector<std::shared_ptr<Link>>& path2) {
  for (const auto& l1 : path1)
    for (const auto& l2 : path2)
      if (l1 == l2) return true;
  return false;
}

void populate_global_to_local_ids(network_group_t& group) {
  int local_id = 0;
  for (auto global_gpu : group.connected_gpus) {
    group.global_to_local_ids[global_gpu] = local_id;
    group.local_to_global_ids[local_id] = global_gpu;
    local_id++;
  }
}

void populate_connected_gpus(network_group_t& group, Topology* topo, bool print) {
  std::string res;
  if ((*group.devices.begin())->type == NODE_GPU) {
    for (const auto& device : group.devices) {
      res += device->toStr() + ",";
      group.connected_gpus.insert(device->local_id);
      group.device_to_connected_gpus[device].insert(device);
    }
  } else {
    assert(group.switch_device != nullptr);
    for (auto device : group.devices) {
      assert(topo->topo_info.switch_device_groups.find(device) != topo->topo_info.switch_device_groups.end());
      auto layer_group = topo->topo_info.switch_device_groups[device];
      for (auto gpu : topo->topo_info.layer_info[layer_group.first].groups[layer_group.second].connected_gpus) {
        group.connected_gpus.insert(gpu);
        group.device_to_connected_gpus[device].insert(topo->npus[gpu].device);
        topo->topo_info.device_groups[topo->npus[gpu].device][group.layer_id] = group.group_id;
        res += topo->npus[gpu].device->toStr() + ",";
      }
    }
  }
  if (group.switch_device != nullptr)
    res += ", connected by " + group.switch_device->toStr();
  if (print)
    LOOOG(ALGO, TRACE, "Layer %d Group %d, %d GPUs: %s", group.layer_id, group.group_id, group.connected_gpus.size(), res.c_str());
}

void generate_intra_group_paths(Topology* topo, network_group_t& group, bool print) {
  std::string res;
  int path_num=0;
  if (group.switch_device == nullptr) {
    for (const auto &it1: group.links) for (const auto &it2: it1.second) {
        group.paths[it1.first][it2.first] = {it2.second};
        path_num++;
        res += it1.first->toStr() + "->" + it2.first->toStr() + ": " + convert_path_to_str(topo, group.paths[it1.first][it2.first]) + "  ";
      }
  } else {
    for (const auto &it1: group.device_to_connected_gpus) for (const auto &it2: group.device_to_connected_gpus) {
        if (it1.first == it2.first) continue;
        group.paths[it1.first][it2.first] = {group.links[it1.first][group.switch_device], group.links[group.switch_device][it2.first]};
        path_num++;
        res += it1.first->toStr() + "->" + it2.first->toStr() + ": " + convert_path_to_str(topo, group.paths[it1.first][it2.first]) + "  ";
      }
    for (const auto &it1: group.device_to_connected_gpus) {
      group.paths[it1.first][group.switch_device] = {group.links[it1.first][group.switch_device]};
      group.paths[group.switch_device][it1.first] = {group.links[group.switch_device][it1.first]};
      path_num += 2;
      res += it1.first->toStr() + "->" + group.switch_device->toStr() + ": " + convert_path_to_str(topo, group.paths[it1.first][group.switch_device]) + "  ";
      res += group.switch_device->toStr() + "->" + it1.first->toStr() + ": " + convert_path_to_str(topo, group.paths[group.switch_device][it1.first]) + "  ";
    }
  }
  if (print)
    LOOOG(ALGO, TRACE, "Layer %d Group %d, %d intra-group paths", group.layer_id, group.group_id, path_num);
}

void generate_cross_layer_gpu_switch_paths(Topology* topo, network_group_t& group, bool print) {
  std::string res;
  int path_num = 0;
  if (group.type == group_switch && (*group.devices.begin())->type != NODE_GPU) {
    for (const auto &it: group.device_to_connected_gpus) {
      auto node = it.first;
      auto lower_layer_group = topo->topo_info.switch_device_groups[node];
      auto lower_group = topo->topo_info.layer_info[lower_layer_group.first].groups[lower_layer_group.second];
      for (auto gpu: it.second) {
        group.paths[gpu][group.switch_device] = merge_paths({lower_group.paths[gpu][node], group.paths[node][group.switch_device]});
        group.paths[group.switch_device][gpu] = merge_paths({group.paths[group.switch_device][node], lower_group.paths[node][gpu]});
        path_num += 2;
        res += gpu->toStr() + "->" + group.switch_device->toStr() + ": " + convert_path_to_str(topo, group.paths[gpu][group.switch_device]) + "  ";
        res += group.switch_device->toStr() + "->" + gpu->toStr() + ": " + convert_path_to_str(topo, group.paths[group.switch_device][gpu]) + "  ";
      }
    }
  }
  if (print)
    LOOOG(ALGO, TRACE, "Layer %d Group %d, %d lower GPU-Switch paths: %s", group.layer_id, group.group_id, path_num, res.c_str());
}

bool is_overlap_with_lower_layer(network_group_t& group, std::shared_ptr<Device> gpu1, std::shared_ptr<Device> gpu2, Topology* topo) {
  for (int layer_id = 0; layer_id < group.layer_id; layer_id++) {
    assert(topo->topo_info.device_groups[gpu1].find(layer_id) != topo->topo_info.device_groups[gpu1].end()
           && topo->topo_info.device_groups[gpu2].find(layer_id) != topo->topo_info.device_groups[gpu2].end());
    if (topo->topo_info.device_groups[gpu1][layer_id] == topo->topo_info.device_groups[gpu2][layer_id]) {
      int group_id = topo->topo_info.device_groups[gpu1][layer_id];
      auto& lower_group_paths = topo->topo_info.layer_info[layer_id].groups[group_id].paths;
      if (lower_group_paths.find(gpu1) != lower_group_paths.end() && lower_group_paths[gpu1].find(gpu2) != lower_group_paths[gpu1].end()) {
        auto lower_group_path = lower_group_paths[gpu1][gpu2];
        if (is_path_overlap(lower_group_path, group.paths[gpu1][group.switch_device]) ||
            is_path_overlap(lower_group_path, group.paths[group.switch_device][gpu2])) {
          return true;
        }
      }
    }
  }
  return false;
}

void generate_cross_layer_inter_gpu_paths(Topology* topo, network_group_t& group, bool print) {
  std::string res;
  int path_num = 0;
  if (group.type == group_switch && (*group.devices.begin())->type != NODE_GPU) {
    for (const auto &it1: group.device_to_connected_gpus)
      for (const auto &it2: group.device_to_connected_gpus) {
        auto node1 = it1.first;
        auto node2 = it2.first;
        if (node1 == node2) continue;
        for (auto gpu1: group.device_to_connected_gpus[node1])
          for (auto gpu2: group.device_to_connected_gpus[node2]) {
            if (!is_overlap_with_lower_layer(group, gpu1, gpu2, topo)) {
              auto group1_pair = topo->topo_info.switch_device_groups[node1];
              auto group2_pair = topo->topo_info.switch_device_groups[node2];
              auto group1 = topo->topo_info.layer_info[group1_pair.first].groups[group1_pair.second];
              auto group2 = topo->topo_info.layer_info[group2_pair.first].groups[group2_pair.second];
              group.paths[gpu1][gpu2] = merge_paths({group1.paths[gpu1][node1], group.paths[node1][node2], group2.paths[node2][gpu2]});
              res += gpu1->toStr() + "->" + gpu2->toStr() + ": " + convert_path_to_str(topo, group.paths[gpu1][gpu2]) + "  ";
              path_num += 1;
            }
          }
      }
  }
  if (print)
    LOOOG(ALGO, TRACE, "Layer %d Group %d, %d lower GPU-GPU paths", group.layer_id, group.group_id, path_num);
}

void generate_layer_paths(Topology* topo, int layer_num) {
  for (int layer_id = 0; layer_id < layer_num; layer_id++) {
    for (int group_id = 0; group_id < topo->topo_info.layer_info[layer_id].groups.size(); group_id++) {
      LOOOG(ALGO, TRACE, "Generating paths for layer %d group %d", layer_id, group_id);
      auto &group = topo->topo_info.layer_info[layer_id].groups[group_id];
      LOOOG(ALGO, TRACE, "Checking group validity...");
      check_group_validity(group);
      bool print = group_id == 0;
      LOOOG(ALGO, TRACE, "Populating connected GPUs...");
      populate_connected_gpus(group, topo, print);
      LOOOG(ALGO, TRACE, "Populating global to local ids...");
      populate_global_to_local_ids(group);
      LOOOG(ALGO, TRACE, "Generating intra group paths...");
      generate_intra_group_paths(topo, group, print);
      LOOOG(ALGO, TRACE, "Generating cross layer paths...");
      generate_cross_layer_gpu_switch_paths(topo, group, print);
      LOOOG(ALGO, TRACE, "Generating cross layer inter GPU paths...");
      generate_cross_layer_inter_gpu_paths(topo, group, print);
    }
  }
}

void clear_layer_paths(Topology* topo, int layer_num) {
  for (int layer_id = 0; layer_id < layer_num; layer_id++) {
    for (auto& group :  topo->topo_info.layer_info[layer_id].groups) {
      auto path_it = group.paths.begin();
      std::string gpu_path_str;
      std::string gpu_str;
      int gpu_path_num = 0;
      while (path_it != group.paths.end()) {
        if (path_it->first->type != NODE_GPU) {
          path_it = group.paths.erase(path_it);
        } else {
          gpu_str += path_it->first->toStr() + ", ";
          auto node_it = path_it->second.begin();
          while (node_it != path_it->second.end()) {
            if (node_it->first->type != NODE_GPU) {
              node_it = path_it->second.erase(node_it);
            } else {
              gpu_path_str += path_it->first->toStr() + "->" + node_it->first->toStr() + ", ";
              gpu_path_num++;
              node_it++;
            }
          }
          path_it++;
        }
      }
      if (group.connected_gpus.size() > 1)
      LOOOG(ALGO, TRACE, "Layer %d Group %d: %d GPUs [%s], %d GPU-GPU paths",
            group.layer_id, group.group_id, group.connected_gpus.size(), gpu_str.c_str(), gpu_path_num);
    }
  }
}


std::pair<layer_relationship_type, layer_relationship_type> layer_relationship(std::set<int>& s1, std::set<int>& s2) {
  if (s1.size() >= s2.size()) {
    for (auto k: s2) {
      if (s1.find(k) == s1.end())
        return {layer_cross, layer_cross};
    }
    return s1.size() == s2.size() ? std::make_pair(layer_repeat, layer_repeat) : std::make_pair(layer_contain, layer_be_contained);
  } else {
    for (auto k : s1) {
      if (s2.find(k) == s2.end())
        return {layer_cross, layer_cross};
    }
    return {layer_be_contained, layer_contain};
  }
}

void print_group_relationship(Topology* topo) {
  for (int i = 0; i < topo->topo_info.layer_info.size(); i++) {
    for (int j = 0; j < topo->topo_info.layer_info.size(); j++) {
      if (i == j) continue;
      std::string res;
      switch (topo->topo_info.layer_info[i].layer_topo_relation[j]) {
        case Simulation::layer_repeat:
          res = "L" + std::to_string(i) + " and L" + std::to_string(j) + " is repeat";
          break;
        case Simulation::layer_contain:
          res = "L" + std::to_string(i) + " contains L" + std::to_string(j) + ", group 0 contains: ";
          for (auto g: topo->topo_info.layer_info[i].groups[0].contained_groups[j])
            res += std::to_string(g) + "/";
          break;
        case Simulation::layer_be_contained:
          res = "L" + std::to_string(i) + " is contained by L" + std::to_string(j);
          break;
        default:
          res = "L" + std::to_string(i) + " and L" + std::to_string(j) + " is cross";
      }
      LOOOG(ALGO, TRACE, "layer: %s", res.c_str());
    }
  }
}

void generate_group_relationship(Topology* topo) {
  // Check whether i contains j
  for (int i = 0; i < topo->topo_info.layer_info.size(); i++) {
    for (int j = i+1; j < topo->topo_info.layer_info.size(); j++) {
      for (int gi = 0; gi < topo->topo_info.layer_info[i].groups.size(); gi++) {
        for (int gj = 0; gj < topo->topo_info.layer_info[j].groups.size(); gj++) {
          std::set<int> group_i = topo->topo_info.layer_info[i].groups[gi].connected_gpus;
          std::set<int> group_j = topo->topo_info.layer_info[j].groups[gj].connected_gpus;
          auto r = layer_relationship(group_i, group_j);
          topo->topo_info.layer_info[i].layer_topo_relation[j] = r.first;
          topo->topo_info.layer_info[j].layer_topo_relation[i] = r.second;
          if (r.first == layer_contain)
            topo->topo_info.layer_info[i].groups[gi].contained_groups[j].push_back(gj);
          else if (r.first == layer_be_contained)
            topo->topo_info.layer_info[j].groups[gj].contained_groups[i].push_back(gi);
        }
      }
    }
  }
  print_group_relationship(topo);
}

void Topology::generateLayerInfo() {
  LOOOG(ALGO, TRACE, "Generate Layer Info");
  int layer_num = topo_info.layer_info.size();
  LOOOG(ALGO, TRACE, "Generating layer paths...");
  generate_layer_paths(this, layer_num);
  LOOOG(ALGO, TRACE, "Clearing layer paths...");
  clear_layer_paths(this, layer_num);
  LOOOG(ALGO, TRACE, "Generating group relationship...");
  generate_group_relationship(this);
}

int topo_info_t::get_layer_num_links(int layer) {
  assert(layer_info.size() > layer);
  assert(!layer_info[layer].groups.empty());
  assert(!layer_info[layer].groups[0].links.empty());
  int nl = 0;
  for (auto &[_, outs] : layer_info[layer].groups[0].links) {
    for (auto &[_, link] : outs) {
      if (link != nullptr) {
        nl++;
      }
    }
  }
  assert(nl % 2 == 0); // bidirectional links
  return nl / 2 * layer_info[layer].groups.size();
}

Bandwidth topo_info_t::get_layer_total_bandwidth(int layer) {
  assert(layer_info.size() > layer);
  assert(!layer_info[layer].groups.empty());
  assert(!layer_info[layer].groups[0].links.empty());
  assert(!layer_info[layer].groups[0].links.begin()->second.empty());
  return layer_info[layer].groups[0].links.begin()->second.begin()->second->getBandwidthBpns() * get_layer_num_links(layer);
}

// TODO: Accurate calculation for direct-connect topologies
Bandwidth topo_info_t::get_layer_avg_bandwidth(int layer) {
  return get_layer_total_bandwidth(layer) / (layer_info[layer].groups[0].connected_gpus.size() * layer_info[layer].groups.size());
}