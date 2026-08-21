// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/json.hpp"
#include "Sketch/parse.h"
#include "PerfModel/PerfModel.h"
#include "Solver/AlgoSolver.h"
#include "Solver/MIPSolver.h"
#include "Solver/common.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <regex>
#include <unistd.h>

class randomTests {
public:
static std::map<int, int> parse_topo_map(nlohmann::json &map) {
  LOOOG(ALGO, INFO, "Parsing Topo Map %s", map.dump().c_str());
  std::map<int, int> dev_to_gpu;
  auto indices = map["indices"];
  int len = indices.size();
  for (int i = 0; i < len; ++i) {
    std::regex e("GPU(\\d+)");
    std::smatch m;
    std::string dev_name = indices[i];
    if (std::regex_search(dev_name, m, e)) {
      dev_to_gpu[i] = std::stoi(m.str(1));
      LOOOG(ALGO, INFO, "Device %d is GPU %d", i, dev_to_gpu[i]);
    }
  }
  return dev_to_gpu;
}

static void testTECCLParse(std::shared_ptr<Solver::AlgoSolverBase> s, std::shared_ptr<Simulation::PerfModel> perf_model, std::string TECCL_input, std::string TOPO_map, std::string output_path, bool a2a) {
  LOOOG(ALGO, INFO, "Testing TECCL Parse, input: %s, topo: %s, output: %s", TECCL_input.c_str(), TOPO_map.c_str(), output_path.c_str());
  std::ifstream input(TECCL_input);
  if (!input.is_open()) {
    std::cerr << "Failed to open " << TECCL_input << std::endl;
    return;
  }
  std::ifstream input_topo(TOPO_map);
  nlohmann::json config, map;
  input >> config;
  input_topo >> map;
  auto dev_to_gpu = parse_topo_map(map);
  int ngpus = dev_to_gpu.size();
  // AllGather1G1Chunks2EpochType.json
  // 1G/1K/1M
  std::regex e("(\\d+)(G|K|M)(\\d+)Chunks.json");
  std::smatch m;
  std::string filename = TECCL_input;
  if (!std::regex_search(filename, m, e)) {
    assert(false);
  }
  int size = std::stoi(m.str(1));
  std::string unit = m.str(2);
  int nchunks = std::stoi(m.str(3));
  Solver::ChunkSize chunk_size_B = unit == "G" ? (1 << 30) : (unit == "K" ? (1 << 10) : (1 << 20));
  chunk_size_B = chunk_size_B * size / ngpus / nchunks;
  auto sol = a2a ? parse_teccl_solution_a2a(config, dev_to_gpu, chunk_size_B, s->topo) :
                   parse_teccl_solution(config, dev_to_gpu, chunk_size_B, s->topo);
  LOOOG(PERFMODEL, INFO, "Solution: %s", sol->toStr().c_str());
  Simulation::SimConfig sim_config;
  sim_config.sol = sol;
  auto res = perf_model->runAnalysis(sim_config);
  LOOOG(PERFMODEL, INFO, "Simulation Result: %s", res->toStr().c_str());
  auto json_output = res->to_json();
  std::ofstream output(output_path);
  output << json_output.dump(2) << std::endl;
}

static void getRings(int ngpus, int machine_size, int nrings, std::vector<std::vector<int>> &rings) {
  assert(machine_size % nrings == 0);
  assert(ngpus % machine_size == 0);
  int offset = machine_size / nrings;
  int nmachines = ngpus / machine_size;
  int gpu_per_machine = machine_size;
  // for a 8GPU 4NIC Clos topology, there should be four rings as follows:
  // 0 1 2 ... 7 -> 8 ... 15 | 2 3 ... 0 1 -> 10 11 ... 15 0 1 | ....
  // So that every NIC is used in one ring
  for (int i = 0; i < nrings; ++i) {
    std::vector<int> &ring = rings.emplace_back();
    for (int m = 0; m < nmachines; ++m) {
      for (int j = 0; j < gpu_per_machine; ++j) {
        int gpu = m * gpu_per_machine + (j + i * offset) % gpu_per_machine;
        ring.push_back(gpu);
      }
    }
  }
}

static std::shared_ptr<Solver::SimResult> genRingAlgo(std::shared_ptr<Simulation::PerfModel> perf_model, Solver::ChunkSize chunk_size_B, int nrings, bool pxn) {
  std::vector<std::vector<int>> rings;
  auto topo = perf_model->getTopo();
  int ng = topo->get_npus_count();
  int machine_size = topo->topo_info.layer_info[1].groups[0].connected_gpus.size();
  assert(machine_size == 8);
  getRings(ng, machine_size, nrings, rings);
  Simulation::SimConfig sim_config;
  assert(chunk_size_B % rings.size() == 0);
  auto sol = std::make_shared<Solver::EventSolution>();
  for (int ringid = 0; ringid < rings.size(); ++ringid) {
    auto &ring_order = rings[ringid];
    for (int i = 0; i < ng; ++i) {
      Solver::SrcChunk chunk = {i, ringid};
      sol->addInitDev(chunk, topo->getNpuDevice(i)->get_id());
      sol->setChunkSize(chunk, chunk_size_B / rings.size());
    }
    for (int step = 0; step < ng - 1; ++step) {
      for (int i = 0; i < ng; ++i) {
        int src = ring_order[i];
        int dst = ring_order[(i + 1) % ng];
        int chunk_id = ring_order[(i + ng - step) % ng];
        Solver::SrcChunk chunk = {chunk_id, ringid};
        sol->addMILPEvent(chunk, src, dst, step);
      }
    }
  }
  sol->sortEvents();
  sim_config.sol = sol;
  sim_config.PXN = pxn;
  LOOOG(PERFMODEL, TRACE, "Simulation Config: %s", sim_config.sol->toStr().c_str());
  auto res = perf_model->runAnalysis(sim_config);
  return res;
}

static std::shared_ptr<Solver::SimResult> genDirectAlgo(std::shared_ptr<Simulation::PerfModel> perf_model, Solver::ChunkSize chunk_size_B, bool pxn) {
  auto topo = perf_model->getTopo();
  int ng = topo->get_npus_count();
  int machine_size = topo->topo_info.layer_info[1].groups[0].connected_gpus.size();
  assert(machine_size == 8);
  Simulation::SimConfig sim_config;
  auto sol = std::make_shared<Solver::EventSolution>();
  for (int i = 0; i < ng; ++i) {
    Solver::SrcChunk chunk = {i, 0};
    sol->addInitDev(chunk, topo->getNpuDevice(i)->get_id());
    sol->setChunkSize(chunk, chunk_size_B);
  }
  for (int step = 0; step < ng - 1; ++step) {
    for (int i = 0; i < ng; ++i) {
      int src = i;
      int dst = (i + step + 1) % ng;
      int chunk_id = i;
      Solver::SrcChunk chunk = {chunk_id, 0};
      sol->addMILPEvent(chunk, src, dst, step);
    }
  }
  sol->sortEvents();
  sim_config.sol = sol;
  sim_config.PXN = pxn;
  LOOOG(PERFMODEL, TRACE, "Simulation Config: %s", sim_config.sol->toStr().c_str());
  auto res = perf_model->runAnalysis(sim_config);
  return res;
}

// ninter allgather for every chunk
static std::shared_ptr<Solver::SimResult> genHierAlgo(std::shared_ptr<Simulation::PerfModel> perf_model, Solver::ChunkSize chunk_size_B, int ninter) {
  auto topo = perf_model->getTopo();
  int ng = topo->get_npus_count();
  int machine_size = topo->topo_info.layer_info[1].groups[0].connected_gpus.size();
  assert(machine_size == 8);
  assert(ng % ninter == 0);
  Simulation::SimConfig sim_config;
  assert(ng % machine_size == 0);
  int nnode = ng / machine_size;
  auto sol = std::make_shared<Solver::EventSolution>();
  for (int i = 0; i < ng; ++i) {
    Solver::SrcChunk chunk = {i, 0};
    sol->addInitDev(chunk, topo->getNpuDevice(i)->get_id());
    sol->setChunkSize(chunk, chunk_size_B);
  }
  // intra-node prepare for inter-node allgather
  int s1step = ninter - 1;
  for (int step = 0; step < s1step; ++step) {
    for (int i = 0; i < ng; ++i) {
      int src = i;
      int chunk_id = i;
      int src_node = i / machine_size;
      for (int j = machine_size / ninter; j < machine_size; j += machine_size / ninter) {
        int dst_node = src_node;
        int dst = dst_node * machine_size + ((i + j) % machine_size);
        Solver::SrcChunk chunk = {chunk_id, 0};
        sol->addMILPEvent(chunk, src, dst, step);
      }
    }
  }
  // inter-node allgather
  int s2step = nnode - 1;
  for (int step = s1step; step < s1step + s2step; ++step) {
    for (int i = 0; i < ng; ++i) {
      int src_node = i / machine_size;
      for (int j = 0; j < machine_size; j += machine_size / ninter) {
        int src = src_node * machine_size + ((i + j) % machine_size);
        int dst_node = (src_node + (step - s1step) + 1) % nnode;
        int dst = dst_node * machine_size + ((i + j) % machine_size);
        int chunk_id = i;
        Solver::SrcChunk chunk = {chunk_id, 0};
        sol->addMILPEvent(chunk, src, dst, step);
      }
    }
  }
  // then intra-node transfer
  int s3step = (machine_size / ninter) - 1;
  for (int step = s1step + s2step; step < s1step + s2step + s3step; ++step) {
    for (int i = 0; i < ng; ++i) {
      for (int src_node = 0; src_node < nnode; ++src_node) {
        for (int j = 0; j < machine_size; j += machine_size / ninter) {
          int src = src_node * machine_size + ((i + j) % machine_size);
          int dst_node = src_node;
          int dst = dst_node * machine_size + ((i + j + step - s1step - s2step + 1) % machine_size);
          int chunk_id = i;
          Solver::SrcChunk chunk = {chunk_id, 0};
          sol->addMILPEvent(chunk, src, dst, step);
        }
      }
    }
  }
  sol->sortEvents();
  sim_config.sol = sol;
  LOOOG(PERFMODEL, TRACE, "Simulation Config: %s", sim_config.sol->toStr().c_str());
  auto res = perf_model->runAnalysis(sim_config);
  return res;
}

static void ncclSim(std::shared_ptr<Simulation::PerfModel> perf_model, Solver::ChunkSize chunk_size_B, std::string output_path) {
  nlohmann::json final_output;
  std::vector<double> alg_times;
  std::vector<std::string> alg_names;
  final_output["algorithms"] = nlohmann::json::array();
  std::vector<std::shared_ptr<Solver::SimResult>> results;
  auto ngpus = perf_model->getTopo()->get_npus_count();
  for (auto pxn : {false, true}) {
    // Workaround, Must have PXN for rail as we didn't include top-layer switch
    if (!pxn && perf_model->getTopo()->topo_info.layer_info.size() < 5) {
      continue;
    }
    for (auto nrings : {1, 2, 4, 8}) {
      LOOOG(PERFMODEL, INFO, "Simulating Ring Allgather with %d rings and PXN %s", nrings, pxn ? "enabled" : "disabled");
      results.push_back(genRingAlgo(perf_model, chunk_size_B, nrings, pxn));
      alg_names.push_back(format("%dring-ag-pxn=%s", nrings, pxn ? "yes" : "no"));
    }
    LOOOG(PERFMODEL, INFO, "Simulating Direct Allgather with PXN %s", pxn ? "enabled" : "disabled");
    results.push_back(genDirectAlgo(perf_model, chunk_size_B, pxn));
    alg_names.push_back(format("direct-ag-pxn=%s", pxn ? "yes" : "no"));
  }
  
  for (auto ninter : {1, 2, 4, 8}) {
    LOOOG(PERFMODEL, INFO, "Simulating Hierarchical Allgather with %d inter-node Allgather", ninter);
    results.push_back(genHierAlgo(perf_model, chunk_size_B, ninter));
    alg_names.push_back(format("hier-ag-ninter=%d", ninter));
  }
  for (int i = 0; i < results.size(); ++i) {
    auto sched = results[i];
    nlohmann::json json = {
      {"chunk_size_B", chunk_size_B},
      {"final_schedule", sched->to_json()},
    };
    auto final_time = sched->time_us;
    alg_times.push_back(final_time);
    final_output["algorithms"].push_back(json);
    LOOOG(PERFMODEL, INFO, "Algorithm: %s, Time: %.3lf us", alg_names[i].c_str(), final_time);
  }
  final_output["alg_times"] = alg_times;
  final_output["sketch_solve_best_time"] = *std::min_element(alg_times.begin(), alg_times.end());
  final_output["num_algs"] = results.size();
  final_output["coll_name"] = "allgather";
  final_output["ngpus"] = perf_model->getTopo()->get_npus_count();
  final_output["chunk_size_byte"] = chunk_size_B;
  std::ofstream output(output_path);
  output << final_output.dump(2) << std::endl;
}

static void reSim(std::shared_ptr<Solver::AlgoSolverBase> s, std::shared_ptr<Simulation::PerfModel> perf_model, std::string file_input, std::string output_path) {
  LOOOG(ALGO, INFO, "Re-simulating %s", file_input.c_str());
  std::ifstream input(file_input);
  if (!input.is_open()) {
    std::cerr << "Failed to open " << file_input << std::endl;
    return;
  }
  nlohmann::json final_output;
  input >> final_output;

  std::string coll_name = final_output["coll_name"];
  double best_time_us = 0;
  nlohmann::json json_output;
  for (auto algo : final_output["algorithms"]) {
    auto schedule = algo["final_schedule"]["Schedule"];
    Simulation::SimConfig sim_config;
    sim_config.sol = Solver::EventSolution::from_json(schedule);
    std::vector<double> times;
    std::vector<std::shared_ptr<Solver::SimResult>> results;
    for (int i = 0; i < 1; ++i) {
      auto res = perf_model->runAnalysis(sim_config);
      results.push_back(res);
      times.push_back(res->time_us);
    }
    for (auto time : times) {
      LOOOG(ALGO, INFO, "Time: %.3lf us", time);
    }
    int min_idx = std::min_element(times.begin(), times.end()) - times.begin();
    auto res = results[min_idx];
    double time_us = res->time_us;
    if (time_us < best_time_us || best_time_us == 0) {
      best_time_us = time_us;
      json_output = res->to_json();
    }
  }
  std::ofstream output(output_path);
  output << json_output.dump(2) << std::endl;
}

};