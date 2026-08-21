// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"

/*
std::queue<CommGraph*> initialize_graph_queue(Collective* coll, ec_mapping_t& ec_mapping, int layer, int e_gpu, std::vector<int>& gpus_c) {
  auto gpus_set = convert_vector_to_set(gpus_c);
  assert(coll->sender_gpus.size() == 1);
  int sender = *coll->sender_gpus.begin();
  auto chunk = coll->getChunk(e_gpu, e_gpu == sender);
  std::queue<CommGraph*> graph_queue;
  auto ring_graph = new CommGraph(ec_mapping, gpus_set, chunk);
  ring_graph->per_layer_algo[layer] = algo_one_group_ring;
  auto tree_graph = new CommGraph(ec_mapping, gpus_set, chunk);
  tree_graph->per_layer_algo[layer] = algo_one_group_tree;
  graph_queue.push(ring_graph);
  graph_queue.push(tree_graph);
  return graph_queue;
}

bool has_path_in_layer(int gpu1, int gpu2, std::shared_ptr<Simulation::Topology> topo, int layer_id) {
  auto node1 = topo->npus[gpu1].device;
  auto node2 = topo->npus[gpu2].device;
  assert(topo->layer_info.device_groups[node1][layer_id] == topo->layer_info.device_groups[node2][layer_id]);
  int group_id = topo->layer_info.device_groups[node1][layer_id];
  auto& paths = topo->layer_info.layer_info[layer_id][group_id].paths;
  return paths.find(node1) != paths.end() && paths[node1].find(node2) != paths[node1].end();
}

bool path_exist_in_layer(std::map<CommGraphNode*, std::vector<int>>& all_mappings, std::shared_ptr<Simulation::Topology> topo, int layer) {
  for (auto& it : all_mappings) {
    for (auto next_gpu : it.second) {
      if (!has_path_in_layer(it.first->chunk.gpu_id, next_gpu, topo, layer))
        return false;
      }
  }
  return true;
}

std::vector<CommGraph *> do_tree_search(CommGraph* current_graph, Collective* coll, std::shared_ptr<Simulation::Topology> topo, int layer, int out_degree) {
  std::vector<CommGraph *> res;
  int remained_gpu_num = current_graph->tmp_remained_gpus.size();
  int end_node_num = current_graph->end_nodes.size();
  std::vector<std::vector<int>> current_layer_gpus;
  if (end_node_num * out_degree <= remained_gpu_num) {
    current_layer_gpus.push_back(convert_set_to_vector(current_graph->tmp_remained_gpus));
  } else {
    current_layer_gpus = find_all_combinations(current_graph->tmp_remained_gpus, end_node_num * out_degree,
                                               end_node_num * out_degree);
  }
  for (auto &current_layer_gpu: current_layer_gpus) {
    auto end_nodes = convert_set_to_vector(current_graph->end_nodes);
    auto all_mappings = find_all_splits(current_layer_gpu, end_nodes);
    for (auto mapping: all_mappings) {
      if (!path_exist_in_layer(mapping, topo, layer))
        continue;

      auto new_graph = new CommGraph(current_graph);
      auto node_mappings = new_graph->mergeWithGraph(current_graph);
      for (auto& it: mapping) {
        for (auto next_gpu: it.second) {
          new_graph->addGraphNode(coll->getChunk(next_gpu, false), node_mappings[it.first]);
        }
      }
      res.push_back(new_graph);
    }
  }
  return res;
}

std::vector<CommGraph*> do_ring_search(CommGraph* current_graph, Collective* coll, std::shared_ptr<Simulation::Topology> topo, int layer) {
  std::vector<CommGraph *> res;
  for (auto next_gpu: current_graph->tmp_remained_gpus) {
    auto prev_gpu = (*current_graph->end_nodes.begin())->chunk.gpu_id;
    if (!has_path_in_layer(next_gpu, prev_gpu, topo, layer))
      continue;

    CommGraph *new_graph = new CommGraph(current_graph);
    new_graph->mergeWithGraph(current_graph);
    res.push_back(new_graph);

    new_graph->addGraphNode(coll->getChunk(next_gpu, false), new_graph->gpu_nodes[prev_gpu]);
  }
  return res;
}

std::vector<CommGraph*> search_ec_pair_algo(ec_mapping_t& ec_mapping, int e_gpu, std::vector<int>& c_gpus, int layer,
                                       Collective* coll, std::shared_ptr<Simulation::Topology> topo) {
  std::vector<CommGraph*> res;
  auto tmp_queue = initialize_graph_queue(coll, ec_mapping, layer, e_gpu, c_gpus);
  while (!tmp_queue.empty()) {
    auto current_graph = tmp_queue.front();
    tmp_queue.pop();

    assert(current_graph->per_layer_algo.find(layer) != current_graph->per_layer_algo.end() &&
      current_graph->per_layer_algo[layer] >= algo_one_group_ring && current_graph->per_layer_algo[layer] <= algo_one_group_tree);
    std::vector<CommGraph*> graphs;
#ifdef per_group_algo_ring_enabled
    if (current_graph->per_layer_algo[layer] == algo_one_group_ring)
      graphs = do_ring_search(current_graph, coll, topo, layer);
#endif
#ifdef per_group_algo_tree_enabled
    if (current_graph->per_layer_algo[layer] == algo_one_group_tree && c_gpus.size() >= tree_min_nodes)
      graphs = do_tree_search(current_graph, coll, topo, layer, tree_out_degree);
#endif
    delete current_graph;

    auto merged_graphs = merge_isomorphic_groups(graphs, topo);
    for (auto graph: merged_graphs) {
      if (graph->tmp_remained_gpus.empty()) {
        res.push_back(graph);
      }
      else {
        tmp_queue.push(graph);
      }
    }
  }
  return res;
}

void delete_graphs(std::vector<CommGraph*>& graphs) {
  for (int i = 0; i < graphs.size(); i++)
    delete graphs[i];
}

bool same_algo_in_same_layer(CommGraph* graph1, CommGraph* graph2) {
  for (auto it : graph1->per_layer_algo) {
    int layer_id = it.first;
    if (graph2->per_layer_algo.find(layer_id) != graph2->per_layer_algo.end())
      if (graph2->per_layer_algo[layer_id] != it.second)
        return false;
  }
  return true;
}

void combine_algorithms(std::vector<CommGraph *>& algorithms, std::map<layer_group_t, std::map<int, std::vector<CommGraph*>>>& per_group_algos) {
  std::queue<CommGraph *> q;

  for (auto &layer_group_it: per_group_algos) {
    for (auto &e_it: layer_group_it.second) {
      if (q.empty()) {
        for (auto g: e_it.second)
          q.push(g);
        continue;
      }
      int n = q.size();
      for (int i = 0; i < n; i++) {
        auto prev_graph = q.front();
        q.pop();
        for (auto per_ec_graph: e_it.second) {
          if (!same_algo_in_same_layer(per_ec_graph, prev_graph))
            continue;
          auto new_graph = new CommGraph(per_ec_graph->ec_mapping);
          new_graph->mergeWithGraph(prev_graph);
          new_graph->mergeWithGraph(per_ec_graph);
          q.push(new_graph);
          LOOOG(ALGO, TRACE, "Merge [%s], [%s] -> [%s]", prev_graph->toStr().c_str(), per_ec_graph->toStr().c_str(), new_graph->toStr().c_str());
        }
        delete prev_graph;
      }
      delete_graphs(e_it.second);
    }
  }
  while (!q.empty()) {
    auto graph = q.front();
    q.pop();
    algorithms.push_back(graph);
  }
}

void clear_algorithms(std::map<layer_group_t, std::map<int, std::vector<CommGraph*>>>& per_group_e_algos) {
  for (auto& it : per_group_e_algos) {
    for (auto& it1 : it.second) {
      for (int i = 0; i < it1.second.size(); i++)
        delete it1.second[i];
    }
  }
}

void search_per_chunk_algo(std::vector<CommGraph*>& graphs, ec_mapping_t& ec_mapping, Collective* coll, std::shared_ptr<Simulation::Topology> topo) {
  std::map<layer_group_t, std::map<int, std::vector<CommGraph*>>> per_group_e_algos;
  for (auto& it_e : ec_mapping.ec_mappings) {
    int gpu_e = it_e.first;
    for (auto& it_layer_group : it_e.second) {
      auto layer_group = it_layer_group.first;
      auto gpus_c = it_layer_group.second;
      auto algos = search_ec_pair_algo(ec_mapping, gpu_e, gpus_c, layer_group.first, coll, topo);
      if (algos.empty()) {
        clear_algorithms(per_group_e_algos);
        return;
      }
      per_group_e_algos[it_layer_group.first][gpu_e] = algos;
    }
  }
  combine_algorithms(graphs, per_group_e_algos);
}

std::vector<CommGraph_t> Algorithm::searchAlgorithms(std::vector<ec_mapping_t>& ec_mappings, Collective* coll) {
  construct_graph_num = 0;
  LOOOG(ALGO, INFO, "Search per-chunk algorithms...");
  std::vector<CommGraph_t> graphs;
  for (int i = 0; i < ec_mappings.size(); i++) {
    LOOOG(ALGO, INFO, "Search algorithms for EC mappings %d/%d %s", i, ec_mappings.size(), ec_mappings[i].toStr().c_str());
    search_per_chunk_algo(graphs, ec_mappings[i], coll, perf_model->getTopo());
  }
  LOOOG(ALGO, INFO, "%d per-chunk algorithms generated", graphs.size());
  for (int i = 0; i < graphs.size(); i++) {
    LOOOG(ALGO, INFO, "Algorithm %d/%d: %s", i+1, graphs.size(), graphs[i]->toStr().c_str());
  }
  return graphs;
}
*/