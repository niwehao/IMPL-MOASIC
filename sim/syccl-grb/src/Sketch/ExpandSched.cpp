// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/CommGraph.h"
#include "Sketch/common.h"
#include "common/log.h"
#include <cassert>
#include <memory>

int sched_graph_dup_t::total_dup = 0;

std::string intra_group_sched::toStr() {
  std::string res;
  for (auto &[chunk_id, sends] : src_dest_pairs) {
    res += format("Chunk#%d:", chunk_id);
    auto &[srcs, dsts] = sends;
    res += format("%s -> %s", setToStr(srcs).c_str(), setToStr(dsts).c_str());
    res += "\n";
  }
  return res;
}

std::string full_sched_graph_node_t::toStr() {
  std::string res;
  res += format("Node#%d, step=%d, layer=%d\n", id, step, layer);
  for (auto &[group_id, group_sched] : group_scheds) {
    res += format("Group#%d:\n%s", group_id, group_sched.toStr().c_str());
  }
  res += format("Deps:%s, ", vecToStr(deps).c_str());
  res += format("Next:%s", vecToStr(next).c_str());
  return res;
}

std::string full_sched_graph_t::toStr() {
  std::string res = format("ngpus=%d\n", ngpus);
  for (auto &node : nodes) {
    res += node.toStr() + "\n";
  }
  return res;
}

std::string GpuMapping::toStr() {
  std::string res;
  res += mapToStr(node_map);
  for (auto [layer, group_map] : layer_map) {
    res += format("Layer#%d: ", layer);
    res += mapToStr(group_map);
  }
  return res;
}

std::string sched_graph_dup_t::toStr() {
  std::string res;
  res += format("dup_graph#%d, prev=#%d, ngpus=%d, src_gpu=%d", id, prev_id, ngpus, src_gpu);
  for (int i = 0; i < ngpus; ++i) {
    res += format("\nGraph#%d:\n%s", i, gpu_maps[i].toStr().c_str());
  }
  return res;
}

std::string sched_graph_t::toStr() {
  std::string res;
  res += format("ngpus=%d, src_gpu=%d\n", ngpus, src_gpu);
  for (auto &node : nodes) {
    res += format("Node#%d, step=%d, layer=%d, group=%d: ", node.id, node.step, node.layer, node.group);
    auto &[srcs, dsts] = node.src_dest_pair;
    res += format("%s -> %s, ", setToStr(srcs).c_str(), setToStr(dsts).c_str());
    res += format("Deps:%s, ", vecToStr(node.deps).c_str());
    res += format("Next:%s\n", vecToStr(node.next).c_str());
  }
  return res;
}

sched_graph_t::sched_graph_t(sched_t &sched) {
  ngpus = sched.gpu_num, src_gpu = sched.sender;
  received_node.resize(ngpus, -1);
  // These are for translating a "broadcast" sketch to "scatter" sketch for alltoall
  // scatter semantic requires the amount of each transition step as comunication is no longer simple copy
  a2a_parent.resize(ngpus);
  a2a_rem_size.resize(ngpus, 0);
  LOG(ALGO, TRACE, "Sched: %s", sched.toStr().c_str()); 
  for (auto &[step_id, step] : sched.comm_by_step) {
    for (auto &[layer_id, layer] : step.layer_group_comm) {
      for (auto &[group_id, group] : layer) {
        assert(group.src_dest_pairs.size() == 1);
        auto &src_dest_pair = group.src_dest_pairs[0];
        int id = nodes.size();
        nodes.emplace_back(id, src_dest_pair, step_id, layer_id, group_id);
        auto &[srcs, dsts] = src_dest_pair;
        for (auto dst : dsts) {
          // every dst receives their data in only one step(or node here)
          assert(received_node[dst] == -1);
          received_node[dst] = id;
        }
        std::set<int> node_deps;
        for (auto src : srcs) {
          if (src != src_gpu) {
            assert(received_node[src] != -1);
            node_deps.insert(received_node[src]);
          }
        }
        for (auto dep : node_deps) {
          nodes[id].deps.emplace_back(dep);
          nodes[dep].next.emplace_back(id);
        }
      }
    }
  }
  extractA2AInfo();
}

sched_graph_t::sched_graph_t(nlohmann::json &sched) {
  ngpus = sched["ngpus"], src_gpu = sched["src_gpu"];
  received_node.resize(ngpus, -1);
  a2a_parent.resize(ngpus);
  a2a_rem_size.resize(ngpus, 0);
  for (auto node : sched["nodes"]){
    int id = node["id"];
    int step = node["step"];
    int group = node["group"];
    int layer = node["layer"];
    std::pair<std::set<int>, std::set<int>> src_dest_pair;
    std::set<int> srcs, dsts;
    for (int src : node["src_dest_pair"]["srcs"]){
      srcs.insert(src);
    }
    for (int dst : node["src_dest_pair"]["dsts"]){
      dsts.insert(dst);
    }
    src_dest_pair = {srcs, dsts};
    sched_graph_node_t graph_node(id, src_dest_pair, step, layer, group);
    for (int dep: node["deps"]){
      graph_node.deps.push_back(dep);
    }
    for (int next: node["next"]){
      graph_node.next.push_back(next);
    }
    nodes.push_back(graph_node);
  }
  extractA2AInfo();
}

void sched_graph_t::extractA2AInfo(){
  // figure out routing using reverse topological order
  std::vector<int> out_deg(nodes.size(), 0);
  std::queue<int> q;
  for (auto &node : nodes) {
    out_deg[node.id] = node.next.size();
    if (out_deg[node.id] == 0) {
      q.push(node.id);
    }
  }
  while (!q.empty()) {
    auto id = q.front(); q.pop();
    auto node = nodes[id];
    auto &[srcs, dsts] = node.src_dest_pair;
    std::vector<int> src_r;
    for (auto src : srcs) {
      src_r.emplace_back(src);
    }
    int nsrc = 0;
    /*
    for multiple srcs sending, broadcast remains the same as every src holds the root chunk
    However, in a scatter formulation, we need to know which chunks the srcs hold in the previous step
    therefore, which chunk is assigned to which src in the current step should be determined before solving this step
    so we implement a simple evenly distributing strategy here, but idealy a topology-aware strategy is preferred
    TODO: or we should search this distribution during sketch exploration instead as this is basically a "routing desicion"
    and alltoall is mostly solving for routing decisions
    */
    for (auto dst : dsts) {
      a2a_rem_size[dst] += 1;
      a2a_parent[dst] = src_r[nsrc];
      a2a_rem_size[src_r[nsrc]] += a2a_rem_size[dst];
      nsrc = (nsrc + 1) % srcs.size();
    }
    for (auto dep : nodes[id].deps) {
      out_deg[dep]--;
      if (out_deg[dep] == 0) {
        q.push(dep);
      }
    }
  }
  for (int i = 0; i < ngpus; ++i) {
    LOOOG(ALGO, TRACE, "a2a parent for GPU#%d: %d", i, a2a_parent[i]);
    LOOOG(ALGO, TRACE, "a2a rem size for GPU#%d: %d", i, a2a_rem_size[i]);
  }
}

ExtractedLayerInfo::ExtractedLayerInfo(Simulation::topo_info_t &topo_info, std::set<int> used_layers): used_layers(used_layers) {
  int ngpus = -1;
  for (auto &layer : topo_info.layer_info) {
    for (auto &[other, rel] : layer.layer_topo_relation) {
      layer_relation[layer.groups[0].layer_id][other] = rel;
    }
    int cnt = 0;
    for (auto &group : layer.groups) {
      int layer_id = group.layer_id, group_id = group.group_id;
      group_size[layer_id][group_id] = group.connected_gpus.size();
      local_to_global[layer_id][group_id] = group.local_to_global_ids;
      global_to_local[layer_id][group_id] = group.global_to_local_ids;
      assert(group.connected_gpus.size() == group.global_to_local_ids.size());
      cnt += group.connected_gpus.size();
      for (auto gpu : group.connected_gpus) {
        group_idx[gpu][layer_id] = {group_id, group.global_to_local_ids[gpu]};
      }
    }
    if (ngpus != -1) {
      assert(ngpus == cnt);
    }
    ngpus = cnt;
  }
  // Add layers without physical switching nodes
  // TODO: For Direct-connect topologies, we need to add all layers that doesn't contain physical switching nodes here
  complete_layer = -1;
  for (auto [layer, group_sizes] : group_size) {
    if (group_sizes.size() == 1 && group_sizes[0] == ngpus) {
      complete_layer = layer;
      used_layers.insert(complete_layer);
    }
  }
  if (complete_layer == -1) {
    complete_layer = add_complete_layer(ngpus);
    used_layers.insert(complete_layer);
  }
  assert(complete_layer != -1);
}

int ExtractedLayerInfo::add_complete_layer(int ngpus) {
  std::set<int> layer_ids;
  for (auto [layer_id, _] : group_size) {
    layer_ids.insert(layer_id);
  }
  int new_layer_id = *std::max_element(layer_ids.begin(), layer_ids.end()) + 1;
  group_size[new_layer_id][0] = ngpus;
  for (int i = 0; i < ngpus; ++i) {
    group_idx[i][new_layer_id] = {0, i};
    local_to_global[new_layer_id][0][i] = i;
    global_to_local[new_layer_id][0][i] = i;
  }
  used_layers.insert(new_layer_id);
  for (auto layer : layer_ids) {
    layer_relation[new_layer_id][layer] = Simulation::layer_contain;
    layer_relation[layer][new_layer_id] = Simulation::layer_be_contained;
  }
  return new_layer_id;
}

std::pair<int, int> ExtractedLayerInfo::getGroupIdx(int gpu, int layer) const {
  try {
    return group_idx.at(gpu).at(layer);
  } catch (std::out_of_range &e) {
    assert(false);
  }
}

std::pair<int, int> ExtractedLayerInfo::getGroupRelIdx(int gpu, int layer, int last_layer) const {
  if (last_layer == -1) {
    return getGroupIdx(gpu, layer);
  } else {
    // get the global group_id in last_layer and translate into local group_id with respect to layer
    auto [group_id, _] = getGroupIdx(gpu, layer);
    auto [last_group_id, __] = getGroupIdx(gpu, last_layer);
    int sz = getLayerRelSize(layer, last_layer);
    return {group_id, last_group_id % sz};
  }
}

int ExtractedLayerInfo::offsetToOtherGroup(int gpu, int other_group, int layer, int x) const {
  auto [group_id, local_id] = getGroupIdx(gpu, layer);
  try {
    int sz = group_size.at(layer).at(group_id);
    int new_local_id = (local_id + sz + x) % sz;
    return local_to_global.at(layer).at(other_group).at(new_local_id);
  } catch (std::out_of_range &e) {
    assert(false);
  }
}

int ExtractedLayerInfo::offsetToRelOtherGroup(int gpu, int other_group, int layer, int last_layer, int x) const {
  if (last_layer == -1) {
    return offsetToOtherGroup(gpu, other_group, layer, x);
  } else {
    auto [group_id, local_id] = getGroupRelIdx(gpu, layer, last_layer);
    try {
      int sz = getLayerRelSize(layer, last_layer);
      int new_local_id = (local_id + sz + x) % sz;
      return other_group * sz + new_local_id;
    } catch (std::out_of_range &e) {
      assert(false);
    }
  }
}

std::set<int> getUsedLayers(const std::shared_ptr<sched_graph_t> &graph) {
  // layer that contains all GPUs may not explicitly exist in the graph(via a switched node etc.)
  // So is not accounted here and is added later.
  std::set<int> used_layers;
  for (auto &node : graph->nodes) {
    used_layers.insert(node.layer);
  }
  return used_layers;
}

/*
Generates a topology-preserving mapping that maps GPU 0 to GPU x
Requires topology layer to be split in a way that every (layer, contained_sub_layer) pair is rotationally symmetric
i.e. for a sub_layer x fully contained in layer y, applying a shift mapping to all groups in layer x that is also contained in layer y preserves the connections between these groups
If the layers are partitioned as such, then this generates a valid topology preserving mapping
For layer partition without such properties, a different method for generating such mapping is required
In theory, multiple valid mappings can exist, but rotational ones are sufficient for our purpose
We need to make sure used_layers contains the top layer containing all gpus
*/
GpuMapping genSingleMapping(int x, ExtractedLayerInfo layer_info, std::set<int> used_layers) {
  // LOOOG(ALGO, INFO, "Generate Single Mapping: x=%d", x);
  std::map<int, int> layer_deg;
  std::map<int, std::vector<int>> layer_contains;
  // Construct a DAG of layers with edges being contain relation, so later we can generate mappings in (reverse)topo order
  for (auto layer : used_layers) {
    // LOOOG(ALGO, INFO, "Used Layer: %d", layer);
    auto group_sizes = layer_info.group_size.at(layer);
    for (auto [group_id, size] : group_sizes) {
      // LOOOG(ALGO, INFO, "Group#%d: %d", group_id, size);
    }
    auto relations = layer_info.layer_relation.at(layer);
    for (auto [other, rel] : relations) {
      if (!used_layers.count(other)) {
        continue;
      }
      // LOOOG(ALGO, INFO, "Layer Relation: %d -> %d: %d", layer, other, rel);
      if (rel == Simulation::layer_contain) {
        layer_contains[layer].push_back(other);
        layer_deg[other]++;
      }
    }
  }
  std::queue<int> q;
  for (auto layer : used_layers) {
    if (layer_deg[layer] == 0) {
      q.push(layer);
    }
  }
  assert(!q.empty());
  // We start from the complete layer, a layer is processed after all layer containing it are processed
  int root_layer = q.front();
  int n = layer_info.group_size.at(root_layer).at(0);
  std::map<int, int> layer_group_offset;
  std::map<int, int> layer_idx_offset;
  std::vector<int> layer_order;
  std::map<int, int> prev_layer;
  while (!q.empty()) {
    int cur_layer = q.front(); q.pop();
    layer_order.push_back(cur_layer);
    // For a complete layer, the mapping is initialized as a direct rotational mapping, shifting by x for 0->x
    // Otherwise it gives the implied intra-group index offset form a parent layer
    int cur_offset = layer_idx_offset.count(cur_layer) ? layer_idx_offset[cur_layer] : x;
    for (auto nxt_layer : layer_contains[cur_layer]) {
      if (!used_layers.count(nxt_layer)) {
        continue;
      }
      layer_deg[nxt_layer]--;
      if (layer_deg[nxt_layer] == 0) {
        /*
        For a complete layer, this gives the group and intra-group index of 0 and x in the next layer
        the mapping of next contained layer is implied by group index shift g0 -> g1 and intra-group index shift i0 -> i1
        the group index shift is then fixed in this layer, while the intra-group index shift may change if there is more sub-layers
        so we pass the intra-group index offset to the next layer to recursively apply the rotational mapping
        */
        auto [g0, i0] = layer_info.group_idx[0][nxt_layer];
        auto [g1, i1] = layer_info.group_idx[cur_offset][nxt_layer];
        // group level offset w.r.t the parent layer
        layer_group_offset[nxt_layer] = abs(g0 - g1);
        layer_idx_offset[nxt_layer] = abs(i0 - i1);
        prev_layer[nxt_layer] = cur_layer;
        q.push(nxt_layer);
      }
    }
  }
  for (auto layer : used_layers) {
    // LOOOG(ALGO, INFO, "Layer#%d: Group Offset: %d, Idx Offset: %d", layer, layer_group_offset[layer], layer_idx_offset[layer]);
  }
  // Now that we have generated the group and intra-group index offsets for all layers, we can construct the actual mapping by applying the offsets in order
  std::map<int, int> gpu_map;
  std::map<int, std::map<int, int>> group_map;
  for (int i = 0; i < n; ++i) {
    std::map<int, int> shifted_x;
    shifted_x[root_layer] = i;
    for (auto layer : layer_order) {
      if (layer == root_layer) {
        group_map[layer][0] = 0;
        continue;
      }
      int prev = prev_layer[layer];
      // LOOOG(ALGO, INFO, "Layer %d, Prev %d", layer, prev);
      int x = shifted_x[prev];
      auto [gx, ix] = layer_info.getGroupIdx(x, layer);
      int rel_size = layer_info.getLayerRelSize(prev, layer);
      auto [_, gx_loc] = layer_info.getGroupRelIdx(x, prev, layer);
      int new_gx_loc = (gx_loc + layer_group_offset[layer]) % rel_size;
      int new_gx = gx / rel_size * rel_size + new_gx_loc;
      int new_x = layer_info.offsetToOtherGroup(x, new_gx, layer, 0);
      shifted_x[layer] = new_x;
      auto [root_gx, __] = layer_info.getGroupIdx(i, layer);
      if (group_map[layer].find(root_gx) != group_map[layer].end()) {
        assert(group_map[layer][root_gx] == new_gx);
      }
      group_map[layer][root_gx] = new_gx;
      // LOOOG(ALGO, INFO, "layer=%d, x=%d, gx=%d, gx_loc=%d, new_gx=%d, new_x=%d", layer, x, gx, gx_loc, new_gx, new_x);
      if (layer_contains[layer].size() == 0) {
        auto [gx, ix] = layer_info.getGroupIdx(new_x, layer);
        int final_x = layer_info.offsetToOtherGroup(x, gx, layer, layer_idx_offset[layer]);
        if (gpu_map.find(i) != gpu_map.end()) {
          // This assert failing means the layer partition is not compatible with this mapping method
          assert(gpu_map[i] == final_x);
        }
        gpu_map[i] = final_x;
        // LOOOG(ALGO, INFO, "Map %d -> %d", i, final_x);
      }
    }
  }
  GpuMapping ret;
  ret.node_map = gpu_map;
  ret.layer_map = group_map;
  return ret;
}

std::shared_ptr<full_sched_graph_t> convertToFullSchedWithMap(const std::shared_ptr<sched_graph_t> &graph, std::vector<GpuMapping> &gpu_maps) {
  auto full_sched = std::make_shared<full_sched_graph_t>();
  int ngpus = graph->ngpus;
  full_sched->ngpus = ngpus;
  full_sched->src_graph = graph;
  auto a2a_parent = graph->a2a_parent;
  auto a2a_rem_size = graph->a2a_rem_size;
  int src_gpu = graph->src_gpu;
  assert(src_gpu == 0);
  full_sched->root_src_gpu = src_gpu;
  full_sched->gpu_maps = gpu_maps;
  for (auto &node : graph->nodes) {
    full_sched->nodes.emplace_back();
    auto &full_node = full_sched->nodes.back();
    full_node.id = node.id, full_node.step = node.step, full_node.layer = node.layer;
    full_node.deps = node.deps, full_node.next = node.next;
    // For every node, map all (src, dst) pairs in this node for source chunk 0 to other source chunks
    // group_scheds: group id -> chunk_id -> (src, dst) pairs
    for (int i = 0; i < ngpus; ++i) {
      if (i == src_gpu) {
        int group_id = node.group;
        auto &[srcs, dsts] = node.src_dest_pair;
        auto &group_sched = full_node.group_scheds[group_id];
        group_sched.src_dest_pairs[i] = {srcs, dsts};
      } else {
        auto &gmap = gpu_maps[i];
        assert(gmap.layer_map[node.layer].find(node.group) != gmap.layer_map[node.layer].end());
        int group_id = gmap.layer_map[node.layer][node.group];
        auto &group_sched = full_node.group_scheds[group_id];
        auto &[srcs, dsts] = group_sched.src_dest_pairs[i];
        auto &[o_srcs, o_dsts] = node.src_dest_pair;
        for (auto src : o_srcs) {
          srcs.insert(gmap.node_map[src]);
        }
        for (auto dst : o_dsts) {
          dsts.insert(gmap.node_map[dst]);
        }
        if (srcs.size() != o_srcs.size() || dsts.size() != o_dsts.size()) {
          LOOOG(ALGO, INFO, "Failed to map node %d to %d, size mismatch", src_gpu, i);
          return nullptr;
        }
        group_sched.src_dest_pairs[i] = {srcs, dsts};
      }
    }
  }
  // map alltoall routing data
  for (int i = 0; i < ngpus; ++i) {
    if (i == src_gpu) {
      full_sched->a2a_chunk_parent[i] = a2a_parent;
      full_sched->a2a_chunk_rem_size[i] = a2a_rem_size;
    } else {
      auto &gmap = gpu_maps[i];
      full_sched->a2a_chunk_parent[i].resize(ngpus);
      full_sched->a2a_chunk_rem_size[i].resize(ngpus);
      for (int j = 0; j < ngpus; ++j) {
        full_sched->a2a_chunk_parent[i][gmap.node_map[j]] = gmap.node_map[a2a_parent[j]];
        full_sched->a2a_chunk_rem_size[i][gmap.node_map[j]] = a2a_rem_size[j];
      }
    }
    LOOOG(ALGO, TRACE, "a2a parent for Chunk%d: %s", i, vecToStr(full_sched->a2a_chunk_parent[i]).c_str());
    LOOOG(ALGO, TRACE, "a2a rem size for Chunk%d: %s", i, vecToStr(full_sched->a2a_chunk_rem_size[i]).c_str());
  }
  return full_sched;
}

// generate induced rotation map from 0 to all nodes
std::vector<std::shared_ptr<full_sched_graph_t>> dupGraph(const std::shared_ptr<sched_graph_t> &graph, Simulation::topo_info_t &topo_info) {
  auto layer_info = ExtractedLayerInfo(topo_info, getUsedLayers(graph));
  int ngpus = graph->ngpus;
  auto used_layers = layer_info.used_layers;
  std::vector<GpuMapping> gpu_maps;
  for (int i = 0; i < ngpus; ++i) {
    // Try to generate a topology-preserving mapping that maps GPU 0 to GPU i
    // So that a sketch for root chunk 0 can be mapped directly to root chunk i
    auto gmap = genSingleMapping(i, layer_info, used_layers);
    gpu_maps.push_back(gmap);
    // LOOOG(ALGO, INFO, "Mapping for GPU#%d: %s", i, gmap.toStr(true).c_str());
  }
  // Generate the sketch containing all root chunks with the mappings
  // Note that for every step, send/recv inside the same group in the same layer (but with possibly different root chunks) is grouped together, and are expected to be solved as one subdemand
  auto fs = convertToFullSchedWithMap(graph, gpu_maps);
  fs->used_layers = used_layers;
  return {fs};
}

std::vector<std::shared_ptr<full_sched_graph_t>> Algorithm::expandSketches(std::vector<std::shared_ptr<sched_graph_t>> &scheds) {
  auto &topo_info = perf_model->getTopo()->topo_info;
  std::vector<std::shared_ptr<full_sched_graph_t>> full_scheds;
  for (auto graph : scheds) {
    auto gen_scheds = dupGraph(graph, topo_info);
    assert(gen_scheds.size() == 1); // assume fixed mapping strategy, more mapping strategies may lead to more sketch from one graph
    full_scheds.insert(full_scheds.end(), gen_scheds.begin(), gen_scheds.end());
  }
  return full_scheds;
}