// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/parse.h"
#include "PerfModel/Topology.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "common/log.h"
#include <memory>
#include <regex>

void parse_coll_config(nlohmann::json& config, std::string& coll_name, int& gpu_num, int& sender_gpu, int& receiver_gpu, uint64_t& byte) {
  auto coll_config = config["coll"];
  coll_name = coll_config["name"];
  sender_gpu = coll_config["root_sender"];
  receiver_gpu = coll_config["root_receiver"];
  byte = coll_config["byte"];
  gpu_num = int(config["hosts"]["host_num"]) * int(config["hosts"]["host_gpu_num"]);
}

void parse_prune_config(nlohmann::json& config, prune_policy_t& prune) {
  for (auto& [key, value] : config.items()) {
    // Step 1
    if (key == "max_comb_layer_num") {
      prune.max_comb_layer_num = int(value);
    } else if (key == "ignored_layers") {
      for (int layer : value)
        prune.ignored_layers.push_back(int(layer));
    } else if (key == "must_contain_layers") {
      for (int layer : value)
        prune.must_contain_layers.push_back(int(layer));
    }
    // Step 2
    else if (key == "layer_comm_ordering") {
      for (auto& [layer_more_comm, layers_less_comm]: value.items()) {
        for (auto layer_less_comm : layers_less_comm) {
          prune.layer_comm_ordering[atoi(layer_more_comm.c_str())].insert(int(layer_less_comm));
        }
      }
    } else if (key == "layer_comm_uplimit") {
      for (auto& [layer, uplimit]: value.items()) {
        prune.layer_comm_uplimit[atoi(layer.c_str())] = uplimit;
      }
    } else if (key == "all_groups_used") {
      prune.all_groups_used = bool(value);
    //} else if (key == "prefer_equal_source_per_group") {
      // prune.prefer_equal_source_per_group = bool(value);
    } else if (key == "allow_unequal_source_per_group") {
      prune.allow_unequal_source_per_group = bool(value);
    //} else if (key == "prefer_equal_dest_per_source") {
      // prune.prefer_equal_dest_per_source = bool(value);
    } else if (key == "allow_unequal_dest_per_source") {
      prune.allow_unequal_dest_per_source = bool(value);
    } else if (key == "allow_source_group_special") {
      prune.allow_source_group_special = bool(value);
    }
    // Step 3
    else if (key == "only_next_step_source") {
      // prune.only_next_step_source = bool(value);
    } else if (key == "start_source_diff_layer") {
      prune.start_source_diff_layer = bool(value);
    } else if (key == "partition_source_groups") {
      // prune.partition_source_groups = bool(value);
    } else if (key == "prune_sources_by_compare_steps") {
      prune.prune_sources_by_compare_steps = bool(value);
    } else if (key == "allow_num_groups_arithmetic_per_step") {
      prune.allow_num_groups_arithmetic_per_step = bool(value);
    } else if (key == "allow_source_per_group_arithmetic_per_step") {
      prune.allow_source_per_group_arithmetic_per_step = bool(value);
    } else if (key == "allow_num_groups_geometric_per_step") {
      prune.allow_num_groups_geometric_per_step = bool(value);
    } else if (key == "allow_source_per_group_geometric_per_step") {
      prune.allow_source_per_group_geometric_per_step = bool(value);
    } else if (key == "drop_small_groups") {
      prune.drop_small_groups = bool(value);
    } else if (key == "drop_small_sources") {
      prune.drop_small_sources = bool(value);
    } else if (key == "partition_sources") {
      prune.partition_sources = bool(value);
    } else if (key == "partition_dests") {
      prune.partition_dests = bool(value);
    } else if (key == "prune_dests") {
      prune.prune_dests = bool(value);
    } else if (key == "allow_dest_balanced") {
      prune.allow_dest_balanced = bool(value);
    } else if (key == "allow_dest_centralized") {
      prune.allow_dest_centralized = bool(value);
    } else if (key == "limit_num_steps_ratio") {
      prune.limit_num_steps_ratio = float(value);
    } else if (key == "limit_num_steps") {
      prune.limit_num_steps = int(value);
    } else {
      LOOOG(ALGO, INFO, "Unknown prune config key: %s=%s", key.c_str(), value.dump().c_str());
      assert(0);
    }
  }
}


int parse_topo_layer_num(nlohmann::json& config) {
  return config["topo"].size();
}

std::vector<std::vector<Simulation::DeviceId>> parse_nvlink_matrix(nlohmann::json& config) {
  std::vector<std::vector<Simulation::DeviceId>> matrix;
  matrix.resize(config["host_links"]["nvlink"].size());
  for (Simulation::DeviceId s = 0; s < config["host_links"]["nvlink"].size(); s++) {
    for (int r_id = 0; r_id < config["host_links"]["nvlink"][s].size(); r_id++) {
      matrix[s].push_back(config["host_links"]["nvlink"][s][r_id].get<Simulation::DeviceId>());
    }
  }
  return matrix;
}

std::string parse_topo_layer_type(nlohmann::json& config, int layer_id) {
  return config["topo"][layer_id]["type"];
}

void parse_host_config(nlohmann::json& config, int& machine_num, int& gpu_per_machine, int& nic_per_machine, std::string& link_type) {
  auto host_config = config["hosts"];
  machine_num = host_config["host_num"].get<int>();
  gpu_per_machine = host_config["host_gpu_num"].get<int>();
  nic_per_machine = host_config["host_nic_num"].get<int>();
  link_type = host_config["host_links"];
}

void parse_switch_config(nlohmann::json& config, int layer_id, int& switch_num, std::string& switch_type, std::pair<double, double>& link_spec) {
  switch_num = config["topo"][layer_id]["switch_num"];
  switch_type = config["topo"][layer_id]["switch_topo"];
  std::string link_type = config["topo"][layer_id]["link_spec"];
  link_spec.first = config["link_spec"][link_type]["bw_mbpus"];
  link_spec.second = config["link_spec"][link_type]["lat_us"];
}

std::pair<double, double> parse_link_spec(nlohmann::json& config, int layer_id) {
  std::string link_type = config["topo"][layer_id]["link_spec"];
  return {config["link_spec"][link_type]["bw_mbpus"], config["link_spec"][link_type]["lat_us"]};
}

std::pair<double, double> parse_local_link_spec(nlohmann::json& config) {
  return {config["link_spec"]["memcpy"]["bw_mbpus"], config["link_spec"]["memcpy"]["lat_us"]};
}

std::shared_ptr<Solver::EventSolution> parse_teccl_solution(nlohmann::json& config, std::map<int, int> &teccl_dev_to_gpu, Solver::ChunkSize chunk_size_B, std::shared_ptr<Simulation::Topology> topo) {
  auto sol = std::make_shared<Solver::EventSolution>();
  auto flows = config["7-Flows"];
  for (auto flow: flows) {
    std::string s = flow;
    // "Chunk 0 from 22 traveled over 22->27 in epoch 0 via switches 17"
    std::regex e("Chunk (\\d+) from (\\d+) traveled over (\\d+)->(\\d+) in epoch (\\d+)( via switches (\\d+))?");
    std::smatch m;
    if (std::regex_search(s, m, e)) {
      int src_c_dev = std::stoi(m.str(2));
      int chunk_idx = std::stoi(m.str(1));
      int src_dev = std::stoi(m.str(3));
      int dst_dev = std::stoi(m.str(4));
      int epoch = std::stoi(m.str(5));
      int gpu_id = teccl_dev_to_gpu[src_c_dev];
      Solver::SrcChunk src_c = {gpu_id, chunk_idx};
      int src_gpu = teccl_dev_to_gpu[src_dev], dst_gpu = teccl_dev_to_gpu[dst_dev];
      sol->addMILPEvent(src_c, src_gpu, dst_gpu, epoch);
      sol->setChunkSize(src_c, chunk_size_B);
      sol->addInitDev(src_c, topo->getNpuDevice(gpu_id)->get_id());
    }
  }
  return sol;
}

std::shared_ptr<Solver::EventSolution> parse_teccl_solution_a2a(nlohmann::json& config, std::map<int, int> &teccl_dev_to_gpu, Solver::ChunkSize chunk_size_B, std::shared_ptr<Simulation::Topology> topo) {
  auto sol = std::make_shared<Solver::EventSolution>();
  auto paths = config["8-Chunk paths"];
  int ngpus = teccl_dev_to_gpu.size();
  for (auto [demand, allsends] : paths.items()) {
    std::string s = demand;
    LOOOG(ALGO, INFO, "demand: %s, allsends: %s", demand.c_str(), allsends.dump().c_str());
    // "Demand at 10 for chunk 2 from 11 met by epoch 43"
    std::regex e("Demand at (\\d+) for chunk (\\d+) from (\\d+) met by epoch (\\d+)");
    std::smatch m;
    std::map<Solver::SrcChunk,
      std::map<std::pair<int, int>,
        std::pair<uint64_t, double>>> send_info;
    if (std::regex_search(s, m, e)) {
      int src_c_dev = std::stoi(m.str(3));
      int chunk_idx = std::stoi(m.str(2));
      int dst_c_dev = std::stoi(m.str(1));
      int src_c_gpu_id = teccl_dev_to_gpu[src_c_dev];
      for (auto sends : allsends) {
        LOOOG(ALGO, INFO, "demand: %s, sends: %s", demand.c_str(), sends.dump().c_str());
        Solver::SrcChunk src_c = {src_c_gpu_id, chunk_idx};
        uint64_t real_chunk_size = chunk_size_B;
        for (auto &send : sends) {
          LOOOG(ALGO, INFO, "send: %s", send.dump().c_str());
          uint64_t epoch = send[0];
          std::string ss = send[1];
          // "14->10 with volume 1.0 in epoch 280 via switches 72"
          std::regex ee("(\\d+)->(\\d+) with volume (\\d+\\.\\d+) in epoch (\\d+)( via switches (\\d+))?");
          std::smatch mm;
          if (std::regex_search(ss, mm, ee)) {
            int src_dev = std::stoi(mm.str(1));
            int dst_dev = std::stoi(mm.str(2));
            double volume = std::stod(mm.str(3));
            int src_gpu = teccl_dev_to_gpu[src_dev], dst_gpu = teccl_dev_to_gpu[dst_dev];
            if (send_info[src_c].count({src_gpu, dst_gpu}) == 0) {
              send_info[src_c][{src_gpu, dst_gpu}] = {epoch, volume};
            } else {
              send_info[src_c][{src_gpu, dst_gpu}].first = std::min(send_info[src_c][{src_gpu, dst_gpu}].first, epoch);
              send_info[src_c][{src_gpu, dst_gpu}].second += volume;
            }
          }
        }
      }
    }
    for (auto &[src_c, sends] : send_info) {
      uint64_t real_chunk_size = chunk_size_B;
      for (auto &[src_dst, info] : sends) {
        auto [epoch, volume] = info;
        auto [src_gpu, dst_gpu] = src_dst;
        sol->addLPEvent(src_c, src_gpu, dst_gpu, epoch, volume);
        if (fabs(volume - 1.0) > 1e-2) {
          LOOOG(ALGO, INFO, "WARNING: TECCL SPLITS CHUNK, Volume is not 1.0: %.10lf", volume);
          double portion = volume * chunk_size_B;
          double rem = portion - floor(portion / SLICE_SIZE_B) * SLICE_SIZE_B;
          if (rem > (double)SLICE_SIZE_B / 2) {
            real_chunk_size = ceil(portion / SLICE_SIZE_B) * SLICE_SIZE_B;
          } else {
            real_chunk_size = floor(portion / SLICE_SIZE_B) * SLICE_SIZE_B;
          }
          LOOOG(ALGO, INFO, "Portion: %.10lf, Chunk Size: %lu", portion, real_chunk_size);
          assert(real_chunk_size % SLICE_SIZE_B == 0 && real_chunk_size > 0);
        }
      }
      sol->setChunkSize(src_c, real_chunk_size);
      sol->addInitDev(src_c, topo->getNpuDevice(src_c.first)->get_id());
    }
  }
  return sol;
}

void parse_sketch(nlohmann::json& config, bool& customize_sketch, bool& use_sketch_input, bool& save_sketch, std::string& sketch_path) {
  customize_sketch = config["sketch"]["customize_sketch"];
  use_sketch_input = config["sketch"]["use_sketch_input"];
  save_sketch = config["sketch"]["save_sketch"];
  sketch_path = config["sketch"]["sketch_path"];
}