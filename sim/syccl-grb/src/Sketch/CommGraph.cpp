// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/CommGraph.h"
#include <queue>
#include <deque>
#include <thread>
#include <utility>

CommGraphNode::CommGraphNode(chunk_t chunk) : chunk(chunk) {
  
}

CommGraphNode* CommGraphNode::getPrevNode(int gpu_id) {
  CommGraphNode* node = nullptr;
  if (comm_prev.find(gpu_id) != comm_prev.end()) {
    node = comm_prev[gpu_id];
    assert(prev_nodes.find(node) != prev_nodes.end());
  }
  return node;
}

CommGraphNode* CommGraphNode::getNextNode(int gpu_id) {
  CommGraphNode* node = nullptr;
  if (comm_next.find(gpu_id) != comm_next.end()) {
    node = comm_next[gpu_id];
    assert(next_nodes.find(node) != next_nodes.end());
  }
  return node;
}

void CommGraphNode::connectToNext(CommGraphNode* node) {
  assert(this->getNextNode(node->chunk.gpu_id)== nullptr);
  assert(node->getPrevNode(this->chunk.gpu_id)== nullptr);
  this->comm_next[node->chunk.gpu_id] = node;
  this->next_nodes.insert(node);
  node->comm_next[this->chunk.gpu_id] = this;
  node->prev_nodes.insert(this);
}

std::string CommGraphNode::toStr(bool prev, bool next) {
  std::string res;
  if (prev && !prev_nodes.empty()) {
    for (auto node: prev_nodes) {
      res += std::to_string(node->chunk.gpu_id) + ",";
    }
    res.pop_back();
    res += "->";
  }
  res += std::to_string(chunk.gpu_id);
  if (next && !next_nodes.empty()) {
    res += "->";
    for (auto node: next_nodes) {
      res += std::to_string(node->chunk.gpu_id) + ",";
    }
    res.pop_back();
  }
  return res;
}

CommGraph::~CommGraph() {
  for (auto &it: gpu_nodes) {
    delete it.second;
  }
}

CommGraphNode* CommGraph::addGraphNode(chunk_t chunk, CommGraphNode* comm_prev) {
  if (comm_prev != nullptr)
    assert(gpu_nodes.find(chunk.gpu_id) == gpu_nodes.end());
  CommGraphNode* cur_node;
  if (gpu_nodes.find(chunk.gpu_id) == gpu_nodes.end()) {
    cur_node = new CommGraphNode(chunk);
    tmp_remained_gpus.erase(chunk.gpu_id);
    gpu_nodes[chunk.gpu_id] = cur_node;
  } else {
    cur_node = gpu_nodes[chunk.gpu_id];
  }

  if (comm_prev != nullptr) {
    comm_prev->connectToNext(cur_node);
    if (end_nodes.find(comm_prev) != end_nodes.end())
      end_nodes.erase(comm_prev);
  } else {
    root_nodes.insert(cur_node);
  }
  end_nodes.insert(cur_node);
  return cur_node;
}

std::map<CommGraphNode*, CommGraphNode*> copy_nodes(CommGraph* old_graph, CommGraph* new_graph) {
  std::map<CommGraphNode*, CommGraphNode*> node_mappings;
  for (const auto& it : old_graph->gpu_nodes) {
    int gpu_id = it.first;
    auto node = it.second;
    if (new_graph->gpu_nodes.find(gpu_id) == new_graph->gpu_nodes.end()) {
      auto new_node = new_graph->addGraphNode(node->chunk);
      node_mappings[node] = new_node;
    } else {
      node_mappings[node] = new_graph->gpu_nodes[gpu_id];
    }
  }
  return node_mappings;
}

void copy_edges(CommGraph* old_graph, std::map<CommGraphNode*, CommGraphNode*>& mp) {
  for (const auto &it: old_graph->gpu_nodes) {
    auto old_node = it.second;
    auto new_node = mp[old_node];
    for (auto old_next_node: old_node->next_nodes) {
      auto new_next_node = mp[old_next_node];
      new_node->connectToNext(new_next_node);
    }
  }
}

void update_roots(CommGraph* graph) {
  graph->root_nodes.clear();
  for (auto it : graph->gpu_nodes) {
    if (it.second->prev_nodes.empty())
      graph->root_nodes.insert(it.second);
  }
}

void update_ends(CommGraph* graph) {
  graph->end_nodes.clear();
  for (auto it : graph->gpu_nodes) {
    if (it.second->next_nodes.empty())
      graph->end_nodes.insert(it.second);
  }
}

/*
std::map<CommGraphNode*, CommGraphNode*> CommGraph::mergeWithGraph(CommGraph* old_graph) {
  for (auto it : old_graph->per_layer_algo) {
    int layer_id = it.first;
    basic_algo_e algo_id = it.second;
    assert(per_layer_algo.find(layer_id) == per_layer_algo.end() || per_layer_algo[layer_id] == algo_id);
    per_layer_algo[layer_id] = algo_id;
  }
  auto node_mappings = copy_nodes(old_graph, this);
  copy_edges(old_graph, node_mappings);
  update_roots(this);
  update_ends(this);
  return node_mappings;
}
 */

std::vector<int> convert_set_to_vec(std::set<CommGraphNode*>& nodes) {
  std::vector<int> res;
  for (auto node : nodes)
    res.push_back(node->chunk.gpu_id);
  return res;
}

std::pair<std::set<CommGraphNode*>, std::vector<int>> CommGraph::getNextLayerNodes(std::set<CommGraphNode *> nodes) {
  if (nodes.empty())
    return {root_nodes, convert_set_to_vec(root_nodes)};
  std::set<CommGraphNode*> next_layer;
  std::vector<int> next_layer_gpus;
  for (auto node: nodes) {
    for (auto next_node : node->next_nodes) {
      next_layer.insert(next_node);
      next_layer_gpus.push_back(next_node->chunk.gpu_id);
    }
  }
  return {next_layer, next_layer_gpus};
}

std::string CommGraph::toStr(bool detail) {
  std::string msg = "#" + std::to_string(id) + ":";
  std::set<CommGraphNode*> nodes_visisted;
  std::queue<CommGraphNode*> q;
  for (auto node : root_nodes)
    q.push(node);
  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    if (!node->next_nodes.empty()) {
      msg += node->toStr(false, true) + "|";
      for (auto next_node: node->next_nodes) {
        if (nodes_visisted.find(next_node) == nodes_visisted.end()) {
          nodes_visisted.insert(next_node);
          q.push(next_node);
        }
      }
    }
  }
  if (!msg.empty()) msg.pop_back();
  return msg;
}

