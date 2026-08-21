// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/searchutils.hpp"
#include <algorithm>
#include <set>
#include "common/ctpl.h"
#include "Sketch/parse.h"

std::string sched_t::toStr() const {
  std::string res;
  res += "[#" + std::to_string(id) + "]" + std::to_string(sender) + " -> " + std::to_string(gpu_num) + " GPUs: ";
  for (auto& [layer, conf] : per_layer_size) {
    res += "L" + std::to_string(layer) + "=";
    for (auto& config : conf.config) {
      res += std::to_string(config.used_group_num) + "*[" +
             std::to_string(config.comm_source_per_group) + "->" +
             std::to_string(config.dest_per_group) + "], ";
    }
  }
  if (!comm_by_step.empty()) {
    for (auto& [step, step_comm] : comm_by_step) {
      for (auto& [layer, layer_comm] : step_comm.layer_group_comm) {
        for (auto& [group, group_comm] : layer_comm) {
          for (auto& src_dest_pair : group_comm.src_dest_pairs) {
            res += "\n\t\t\t\t\t\t\t\t\t\tstep=" + std::to_string(step)
                    + ": [L#" + std::to_string(layer) + ", G#" + std::to_string(group)
                    + "] {[" + std::to_string(src_dest_pair.first.size()) + "]";
            for (auto src : src_dest_pair.first)
              res += std::to_string(src) + ",";
            res.pop_back();
            res += "} -> {[" + std::to_string(src_dest_pair.second.size()) + "]";
            for (auto dst : src_dest_pair.second)
              res += std::to_string(dst) + ",";
            res.pop_back();
            res += "}";
          }
        }
      }
    }
  }
  return res;
}

sched_t initialize_scheduling(Collective* collective) {
  sched_t e0;
  // Step 1
  gid = 0;
  e0.id = gid++;
  e0.sender = *collective->sender_gpus.begin();
  assert(std::find(collective->reciever_gpus.begin(), collective->reciever_gpus.end(), e0.sender) != collective->reciever_gpus.end());
  e0.gpu_num = collective->reciever_gpus.size();
  // Step 2
  e0.tmp_acc_dest = 1;
  // Step 3
  e0.cur_step_sources.insert(e0.sender);
  e0.comm_by_gpus[e0.sender] = {-1, -1, -1};
  return e0;
}

std::vector<sched_t> Algorithm::searchEcGpus(Collective* collective) {
  assert(collective->sender_gpus.size()== 1 && collective->reciever_gpus.size() == perf_model->getTopo()->npus.size());
  auto e0 = initialize_scheduling(collective);

/* 
  Step 1: determine which layers an algorithm should use (layer combination)
  for all 2^{num_layers} combinations, filter out all valid ones
  e.g. {L1, L3, L4} / L1=L3=L4=
*/
  auto layer_combs = search_sched_layers(prune, perf_model->getTopo(), e0);

/*
  Step 2: for each layer combination, determine how many groups are used and how many src/dst in each group
  in every used_layer l, n_l group is used, with each group i having m_i srcs and k_i dsts, with srcs being a inital chunk or chunk received in a previous step. A group combination is represented as[m_i->k_i]
  e.g. L1=1*[2->2], L3=1*[8->8], 1*[4->12], L4=1*[9->9]
  The mapping of group configuration to actual groups and the ordering(step number) between group configurations is not determined yet.
*/
  auto layer_sizes = search_sched_groups(prune, layer_combs);

/*
  Step 3: for each group configuration, determine the communication steps
  one group configuration can be used in multiple steps
  e.g. L1=2*[1->7], L3=1*[1->8], L4=1*[9->9], 
  step=0: [L#1, G#0] {[1]0} -> {[7]1,2,3,4,5,6,7}
  step=0: [L#3, G#0] {[1]0} -> {[8]8,9,10,11,12,13,14,15}
  step=0: [L#4, G#0] {[1]0} -> {[1]16}
  step=1: [L#1, G#2] {[1]16} -> {[7]17,18,19,20,21,22,23}
  step=1: [L#4, G#0] {[8]1,2,3,4,5,6,7,8} -> {[8]24,25,26,27,28,29,30,31}
*/
  auto layer_scheds = search_sched_steps(prune, layer_sizes);
  return layer_scheds;
}

void Algorithm::generate_layer_matrix(std::shared_ptr<Simulation::Topology> topo, std::vector<std::vector<int>>& layer_matrix, std::vector<std::vector<int>>& group_matrix){
  int layer_num = topo->topo_info.layer_info.size();
  for (int layer = layer_num - 1; layer >= 0; layer--) {
    if (std::find(prune.must_contain_layers.begin(), prune.must_contain_layers.end(), layer) == prune.must_contain_layers.end()) continue;
    for (int group = 0; group < topo->topo_info.layer_info[layer].groups.size(); group++) {
      auto gpus = topo->topo_info.layer_info[layer].groups[group].connected_gpus;
      for (int src: gpus) {
        for (int dst: gpus){
          layer_matrix[src][dst] = layer;
          group_matrix[src][dst] = group;
        }
      }
    }
  }
}

struct TreeNode {
  int val;
  std::vector<TreeNode*> children;
  TreeNode(int x) : val(x) {}
};

void transfer_to_sketch(TreeNode* root, int step, sched_t& res, std::vector<std::vector<int>> layer_matrix, std::vector<std::vector<int>> group_matrix){
  // Using depth first search algorithm to transfer the tree to a sketch, i.e. construct res.comm_by_step
  for (auto child: root->children){
    int layer = layer_matrix[root->val][child->val];
    int group = group_matrix[root->val][child->val];
    if (res.comm_by_step.find(step)==res.comm_by_step.end() 
    || res.comm_by_step[step].layer_group_comm.find(layer)==res.comm_by_step[step].layer_group_comm.end()
    || res.comm_by_step[step].layer_group_comm[layer].find(group)==res.comm_by_step[step].layer_group_comm[layer].end()){
      std::pair<std::set<int>, std::set<int>> p = {{}, {}};
      res.comm_by_step[step].layer_group_comm[layer][group].src_dest_pairs.push_back(p);
    }
    auto& src_dest_pairs = res.comm_by_step[step].layer_group_comm[layer][group].src_dest_pairs;
    src_dest_pairs[0].first.insert(root->val);
    src_dest_pairs[0].second.insert(child->val);
    transfer_to_sketch(child, step+1, res, layer_matrix, group_matrix);
  }
}

std::vector<sched_t> Algorithm::CustomizeSketch() {
  int machine_num, gpu_per_machine, nic_per_machine;
  std::string host_type;
  parse_host_config(config, machine_num, gpu_per_machine, nic_per_machine, host_type);
  int num_gpus = machine_num * gpu_per_machine;
  
  // matrix record which layer and group two GPUs are located in
  std::vector<std::vector<int>> layer_matrix(num_gpus, std::vector<int>(num_gpus));
  std::vector<std::vector<int>> group_matrix(num_gpus, std::vector<int>(num_gpus));
  generate_layer_matrix(perf_model->getTopo(), layer_matrix, group_matrix);
  
  std::vector<sched_t> sketches; // store the sketches

  /*
  Define your own sketches here.
  Typically, you can construct a tree to represent a sketch.
  Here we provide two examples of 4 machines * 8 gpu_per_machine. 
  (1) GPU0 sends data to GPU8/16/24 and all of them send data to other GPUs in the same machines:
  0 -> {1,2,3,4,5,6,7}
    -> 8 -> {9,10,11,12,13,14,15}
    -> 16 -> {17,18,19,20,21,22,23}
    -> 24 -> {25,26,27,28,29,30,31}
  */
  std::vector<TreeNode*> tree_nodes1(num_gpus);
  for (int i = 0; i < num_gpus; i++){
    tree_nodes1[i] = new TreeNode(i);
  }
  // 0 -> 8/16/24
  for (int i = 1; i < machine_num; i++){
    tree_nodes1[0]->children.push_back(tree_nodes1[i*gpu_per_machine]);
  }
  // 0/8/16/24 to others
  for (int i = 0; i < num_gpus; i += num_gpus/machine_num){
    for (int j = 1; j < gpu_per_machine; j++){
      tree_nodes1[i]->children.push_back(tree_nodes1[i+j]);
    }
  }
  // Transfer the tree to a sketch
  sched_t s1;
  s1.gpu_num = num_gpus;
  s1.sender = 0;
  transfer_to_sketch(tree_nodes1[0], 0, s1, layer_matrix, group_matrix);
  sketches.push_back(s1);
  
  /*
  (2) GPU0 sends data to GPU8/16, GPU16 sends data to GPU24 and all of them send data to other GPUs in the same machines:
  0 -> {1,2,3,4,5,6,7}
    -> 8 -> {9,10,11,12,13,14,15}
    -> 16 -> {17,18,19,20,21,22,23}
          -> 24 -> {25,26,27,28,29,30,31}
  */
  std::vector<TreeNode*> tree_nodes2(num_gpus);
  for (int i = 0; i < num_gpus; i++){
    tree_nodes2[i] = new TreeNode(i);
  }
  // 0 -> 16
  tree_nodes2[0]->children.push_back(tree_nodes2[num_gpus/2]);
  // 0 -> 8, 16 -> 24
  for (int i = 0; i < num_gpus; i += num_gpus/2){
    tree_nodes2[i]->children.push_back(tree_nodes2[i+num_gpus/4]);
  }
  // 0/8/16/24 to others
  for (int i = 0; i < num_gpus; i += num_gpus/4){
    for (int j = 1; j < gpu_per_machine; j++){
      tree_nodes2[i]->children.push_back(tree_nodes2[i+j]);
    }
  }

  sched_t s2;
  s2.gpu_num = num_gpus;
  s2.sender = 0;
  transfer_to_sketch(tree_nodes2[0], 0, s2, layer_matrix, group_matrix);
  sketches.push_back(s2);
  
  return sketches;
}