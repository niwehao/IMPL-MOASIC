// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/CommGraph.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "common/json.hpp"
#include <iostream>
#include <memory>
#include <fstream>
#include <thread>
#include "Sketch/tests.hpp"
#include "common/ctpl.h"

void initTopoAndPerfModel(std::shared_ptr<Simulation::Topology> &topo, std::shared_ptr<Simulation::PerfModel> &perf_model, nlohmann::json &config) {
  topo = Simulation::constructTopology(config);
  perf_model = std::make_shared<Simulation::PerfModel>(topo);
}

Algorithm::Algorithm(nlohmann::json& config, std::shared_ptr<Solver::AlgoSolverBase> solver) : config(config), solver(solver) {
  std::vector<std::thread> threads;
  for (int i = 0; i < MAX_THREAD_NUM; i++) {
    threads.emplace_back(initTopoAndPerfModel, std::ref(topo_pool[i]), std::ref(perf_model_pool[i]), std::ref(config));
  }
  for (auto &th : threads) if (th.joinable()) th.join();
  perf_model = perf_model_pool[0];
}

Algorithm::~Algorithm() {
  delete root_coll;
}

algo_solve_config_t parse_algo_solve_config(const nlohmann::json& config) {
  auto conf = config["algo_solve"];
  algo_solve_config_t solve_config;
  solve_config.num_used_algos_min = conf["num_used_algos_min"];
  solve_config.num_used_algos_max = conf["num_used_algos_max"];
  solve_config.filter_balance_ratio = conf["filter_balance_ratio"];
  solve_config.solve_balance_ratio = conf["solve_balance_ratio"];
  solve_config.time_diff_threshold = conf["time_diff_threshold"];
  solve_config.raw_solve_balance_ratio = conf["raw_solve_balance_ratio"];
  solve_config.do_raw_solve = conf["do_raw_solve"];
  solve_config.solve_output = conf["solve_output"];
  return solve_config;
}

void Algorithm::generate() {
  std::string coll_name;
  int ngpus, sender_gpu, receiver_gpu;
  uint64_t byte;
  parse_coll_config(config, coll_name, ngpus, sender_gpu, receiver_gpu, byte);
  root_coll = new Collective(coll_name, ngpus, byte, sender_gpu, receiver_gpu);
  auto coll_type = root_coll->coll_type;
  bool a2a = coll_type == coll_alltoall;
  // We only solve for all-to-all type collectives for now, for one-to-all we doesn't need to split the root_coll
  auto isomorphic_coll_groups = splitCollectives();
  bool customize_sketch, use_sketch_input, save_sketch;
  std::string sketch_path;
  parse_sketch(config, customize_sketch, use_sketch_input, save_sketch, sketch_path);

  auto solve_config = parse_algo_solve_config(config);
  parse_prune_config(config["prune"], prune);
  Solver::ChunkSize chunk_size_B = byte;

  Clock_ clk;
  // For Symmetric all-to-all, there should only be one group
  assert(isomorphic_coll_groups.size() == 1);
  for (const auto& coll : isomorphic_coll_groups) {
    LOOOG(ALGO, INFO, "Collective: %s", coll->toStr().c_str());
    clk.tic();

    std::ifstream sketch_input(sketch_path);
    nlohmann::json sketch_json;
    std::vector<std::shared_ptr<sched_graph_t>> graph_scheds;

    if (use_sketch_input && sketch_input.is_open()) {
      sketch_input >> sketch_json;    
      for (auto sched : sketch_json){
        graph_scheds.push_back(std::make_shared<sched_graph_t>(sched));
      }  
    }
    else{
      std::vector<sched_t> layer_scheds = customize_sketch ? CustomizeSketch() : searchEcGpus(coll);
      for (auto& sched : layer_scheds) {
        // covert a searched schedule into a graph representation
        // also generating alltoall specific attributes, see comments inside
        graph_scheds.push_back(std::make_shared<sched_graph_t>(sched));
      }
      for (auto &gsched : graph_scheds) {
        LOOOG(ALGO, TRACE, "Generated Single Chunk Sketch: %s", gsched->toStr().c_str());
        sketch_json.push_back(gsched->to_json());
      }
      if (save_sketch){
        std::ofstream sketch_output(sketch_path);
        sketch_output << sketch_json.dump(2);  
      }
    }
    auto ec_search_ms = clk.tic();

    LOOOG(ALGO, INFO, "Step 1: Expand Sketches to All-to-All ones"); clk.tic();
    // Replicate the sketches into all-to-all ones with a layer-level rotational mapping
    // Note that this might not work for invalid layer partition input/topologies
    // In such case, other mapping strategies are required
    auto full_sketches = expandSketches(graph_scheds);
    auto expand_ms = clk.tic();
    for (auto fs : full_sketches) {
      LOOOG(ALGO, INFO, "Generated Full Sketch:\n%s", fs->toStr().c_str());
    }

    LOOOG(ALGO, INFO, "Step 2: Combine Expanded Sketches"); clk.tic();
    // Combine the sketches according to bandwidth usage
    // We only consider two sketch combinations and bandwidth usage of two dimensions now as it's enough for Clos and multi-rail topologies
    auto combined_sketches = combineSketches(full_sketches, a2a);
    auto combine_ms = clk.tic();

    LOOOG(ALGO, INFO, "Step 3: Filter Combined Sketches"); clk.tic();
    // Coarse-grained solving step for the combined sketches
    double s1_balance_ratio = solve_config.filter_balance_ratio;
    double filter_sim_time;
    std::vector<std::shared_ptr<generated_algorithm_t>> filtered_algos = combinedSketchesSolve(combined_sketches, s1_balance_ratio, chunk_size_B, 1, a2a, filter_sim_time);
    std::sort(filtered_algos.begin(), filtered_algos.end(), [](const std::shared_ptr<generated_algorithm_t> &a, const std::shared_ptr<generated_algorithm_t> &b) {
      return a->final_schedule->time_us < b->final_schedule->time_us;
    });
    for (auto algo : filtered_algos) {
      LOOOG(ALGO, INFO, "Sketch: %s", algo->sketch->brief().c_str());
      LOOOG(ALGO, INFO, "Balance Ratio: %.2lf", algo->balance_ratio);
      auto final_time = algo->final_schedule->time_us;
      // LOOOG(ALGO, INFO, "Final Schedule: %s", algo->final_schedule->toStr(false).c_str());
      LOOOG(ALGO, INFO, "Final Simulated Time: %.2lf us", final_time);
    }
    auto min_time = filtered_algos[0]->final_schedule->time_us;
    std::vector<std::shared_ptr<combined_sketch_t>> filtered_sketches;
    for (int i = 0; i < filtered_algos.size(); ++i) {
      const auto &algo = filtered_algos[i];
      if ((double)(algo->final_schedule->time_us - min_time) / min_time < solve_config.time_diff_threshold ||
          i < solve_config.num_used_algos_min) {
        if (i >= solve_config.num_used_algos_max) break;
        filtered_sketches.push_back(filtered_algos[i]->sketch);
      }
    }
    auto filter_ms = clk.tic();

    LOOOG(ALGO, INFO, "Step 4: Fine-Grain Solve Filtered Sketches"); clk.tic();
    // Fine-grained solving step for the remaining filtered sketches
    double s2_balance_ratio = solve_config.solve_balance_ratio;
    double fine_grain_solve_sim_time;
    std::vector<std::shared_ptr<generated_algorithm_t>> generated_algos = combinedSketchesSolve(filtered_sketches, s2_balance_ratio, chunk_size_B, 1, a2a, fine_grain_solve_sim_time);
    std::sort(generated_algos.begin(), generated_algos.end(), [](const std::shared_ptr<generated_algorithm_t> &a, const std::shared_ptr<generated_algorithm_t> &b) {
      return a->final_schedule->time_us < b->final_schedule->time_us;
    });
    auto fine_grain_solve_ms = clk.tic();

    // Record metadata
    nlohmann::json final_output;
    std::vector<double> alg_times;
    for (auto algo : generated_algos) {
      LOOOG(ALGO, INFO, "Sketch: %s", algo->sketch->brief().c_str());
      LOOOG(ALGO, INFO, "Balance Ratio: %.2lf", algo->balance_ratio);
      auto final_time = algo->final_schedule->time_us;
      // LOOOG(ALGO, INFO, "Final Schedule: %s", algo->final_schedule->toStr(false).c_str());
      LOOOG(ALGO, INFO, "Final Simulated Time: %.2lf us", final_time);
      alg_times.push_back(final_time);
      auto json = algo->to_json();
      final_output["algorithms"].push_back(json);
    }
    final_output["coll_name"] = coll_name;
    final_output["ngpus"] = ngpus;
    final_output["chunk_size_byte"] = byte;
    final_output["num_algs"] = generated_algos.size();
    final_output["alg_times"] = alg_times;
    final_output["time_usage"] = {
      {"ec_search_ms", ec_search_ms},
      {"expand_ms", expand_ms},
      {"combine_ms", combine_ms},
      {"filter_ms", filter_ms},
      {"filter_sim_ms", filter_sim_time},
      {"fine_grain_solve_ms", fine_grain_solve_ms},
      {"fine_grain_solve_sim_ms", fine_grain_solve_sim_time}
    };
    final_output["num_lp_solves"] = Solver::MIPSolver::num_lp_solves;
    final_output["num_milp_solves"] = Solver::MIPSolver::num_milp_solves;
    final_output["all_nepoches"] = Solver::MIPSolver::all_nepoches;
    final_output["all_nsolves"] = Solver::MIPSolver::all_nsolves;
    final_output["all_epoch_duration_ns"] = Solver::MIPSolver::all_epoch_duration_ns;
    final_output["all_finish_time_ns"] = Solver::MIPSolver::all_finish_time_ns;
    final_output["all_epoch_probe_time_s"] = Solver::MIPSolver::all_epoch_probe_time;
    final_output["all_construct_time_s"] = Solver::MIPSolver::all_construct_time;
    final_output["all_solve_time_s"] = Solver::MIPSolver::all_solve_time;
    LOOOG(ALGO, INFO, "num_lp_solves: %d, num_milp_solves: %d", Solver::MIPSolver::num_lp_solves, Solver::MIPSolver::num_milp_solves);
    double best_time = generated_algos[0]->final_schedule->time_us;
    LOOOG(ALGO, INFO, "Best Time: %.2lf us", best_time);

    final_output["sketch_solve_best_time"] = best_time;
    std::string dump_path = solve_config.solve_output;
    std::ofstream dump_file(dump_path);
    dump_file << final_output.dump(2);
    // assuming only one group here
    break;
  }
}

