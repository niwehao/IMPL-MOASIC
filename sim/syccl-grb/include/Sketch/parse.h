// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/json.hpp"
#include "PerfModel/Type.h"
#include "Solver/common.h"
#include "Sketch/common.h"

int parse_topo_layer_num(nlohmann::json& config);
std::string parse_topo_layer_type(nlohmann::json& config, int layer_id);
std::vector<std::vector<Simulation::DeviceId>> parse_nvlink_matrix(nlohmann::json& config);
std::pair<double, double> parse_link_spec(nlohmann::json& config, int layer_id);
void parse_host_config(nlohmann::json& config, int& machine_num, int& gpu_per_machine, int& nic_per_machine, std::string& host_type);
void parse_switch_config(nlohmann::json& config, int layer_id, int& switch_num, std::string& switch_type, std::pair<double, double>& link_spec);

void parse_coll_config(nlohmann::json& config, std::string& coll_name, int& gpu_num, int& sender_gpu, int& receiver_gpu, uint64_t& byte);
std::shared_ptr<Solver::EventSolution> parse_teccl_solution (nlohmann::json& config, std::map<int, int> &teccl_dev_to_gpu, Solver::ChunkSize chunk_size_B, std::shared_ptr<Simulation::Topology> topo);
std::shared_ptr<Solver::EventSolution> parse_teccl_solution_a2a(nlohmann::json& config, std::map<int, int> &teccl_dev_to_gpu, Solver::ChunkSize chunk_size_B, std::shared_ptr<Simulation::Topology> topo);
void parse_sketch(nlohmann::json& config, bool& customize_sketch, bool& use_sketch_input, bool& save_sketch, std::string& sketch_path);

void parse_prune_config(nlohmann::json& config, prune_policy_t& prune);
