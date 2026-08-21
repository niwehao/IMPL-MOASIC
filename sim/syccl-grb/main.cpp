/* 
*Copyright (c) 2025, Alibaba Cloud and its affiliates;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include "common/json.hpp"
#include "common/CLI11.hpp"
#include "Sketch/tests.hpp"
#include "Sketch/Algorithm.h"
#include "Solver/MIPSolver.h"
#include "Sketch/parse.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <unistd.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

bool parse_conf_file(std::string config_file, nlohmann::json& config) {
  std::ifstream input(config_file);
  if (!input.is_open()) {
    std::cerr << "Failed to open " << config_file << std::endl;
    return false;
  }
  input >> config;
  if (!config.contains("solver")) {
    std::cerr << "Solver config not found" << std::endl;
    return false;
  }
  if (!config["solver"].contains("mip_config")) {
    std::cerr << "MIP config not found" << std::endl;
    return false;
  }
  return true;
}

int gid = 0;
int SKETCH_EXPLORATION_THREAD_NUM = MAX_THREAD_NUM;
int PARALLEL_THREAD_NUM = MAX_THREAD_NUM / 2;
int PERFMODEL_POOL_SIZE = PARALLEL_THREAD_NUM;
void init_global_stuff() {
  Solver::MIPSolver::num_lp_solves = 0;
  Solver::MIPSolver::num_milp_solves = 0;
  if (std::getenv("SYNTHESIZE_PARALLEL_THREAD_NUM")) {
    PARALLEL_THREAD_NUM = std::stoi(std::getenv("SYNTHESIZE_PARALLEL_THREAD_NUM"));
    PERFMODEL_POOL_SIZE = PARALLEL_THREAD_NUM;
    LOOOG(ALGO, INFO, "Set parallel thread num to %d", PARALLEL_THREAD_NUM);
  }
  if (std::getenv("SKETCH_EXPLORATION_MAX_THREAD_NUM")) {
    SKETCH_EXPLORATION_THREAD_NUM = std::stoi(std::getenv("SKETCH_EXPLORATION_MAX_THREAD_NUM"));
    LOOOG(ALGO, INFO, "Set sketch exploration max thread num to %d", SKETCH_EXPLORATION_THREAD_NUM);
  }
}

int main(int argc, char **argv) {
  init_global_stuff();
  
  std::string config_file = "../config/default.json";
  CLI::App app{"Synthesizer"};
  app.add_option("-f", config_file, "Topology and Collective Config")->default_str(config_file);

  app.require_subcommand(1);
  auto solve = app.add_subcommand("solve", "Solve");
  auto ncclsim = app.add_subcommand("ncclsim", "Test Simulation");
  auto bench = app.add_subcommand("benchmark", "Benchmark TECCL output");
  auto resim = app.add_subcommand("resim", "Re-simulate output with new config");

  // TECCL inputs
  std::string TECCL_input, TOPO_map, output_path;
  bool teccl_a2a = false;
  bench->add_option("-i", TECCL_input, "TECCL output file");
  bench->add_option("--topo", TOPO_map, "Topology mapping file");
  bench->add_option("-o", output_path, "Output path");
  bench->add_flag("--a2a", teccl_a2a, "A2A benchmark");

  std::string solve_res, resim_output;
  resim->add_option("-i", solve_res, "Synthesizer output file");
  resim->add_option("-o", resim_output, "Output path");

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  nlohmann::json config;
  if (!parse_conf_file(config_file, config))
    return 1;

  auto topo = Simulation::constructTopology(config);
  auto perf_model = std::make_shared<Simulation::PerfModel>(topo);
  auto solver = std::make_shared<Solver::MIPSolver>(topo, config["solver"]);

  // Entrypoint for full solve
  if (solve->parsed()) {
    Algorithm algo(config, solver);
    algo.generate();
  }

  if (bench->parsed()) {
    randomTests::testTECCLParse(solver, perf_model, TECCL_input, TOPO_map, output_path, teccl_a2a);
  }

  if (ncclsim->parsed()) {
    std::string coll_name;
    int ngpus, sender_gpu, receiver_gpu;
    uint64_t byte;
    parse_coll_config(config, coll_name, ngpus, sender_gpu, receiver_gpu, byte);
    auto dump_path = config["algo_solve"]["solve_output"].get<std::string>();
    dump_path = dump_path.substr(0, dump_path.find_last_of(".")) + "-static.json";
    randomTests::ncclSim(perf_model, byte, dump_path);
  }

  if (resim->parsed()) {
    randomTests::reSim(solver, perf_model, solve_res, resim_output);
  }

  return 0;
}
