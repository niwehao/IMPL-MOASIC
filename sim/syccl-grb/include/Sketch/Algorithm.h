// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/json.hpp"
#include "Sketch/CommGraph.h"
#include "Sketch/Collective.h"
#include "Sketch/common.h"
#include "PerfModel/PerfModel.h"
#include "PerfModel/Topology.h"
#include "Solver/MIPSolver.h"
#include "Solver/common.h"
#include <memory>
#include <set>
#include <vector>

using subdemand_hash_t = std::pair<int, std::map<std::pair<int, int>, int>>;
// (layer, split_chunks, balance_ratio, chunk_size)
using solver_param_key_t = std::tuple<int, int, double, uint64_t>;
class SketchSubDemand {
public:
    int layer_id, group_id;
    intra_group_sched *sched;
    int split_chunks;
    double balance_ratio;
    uint64_t chunk_size_B;
    std::set<int> used_layers;
    // for a2a
    std::map<int, std::vector<int>> *a2a_chunk_parent;
    std::map<int, std::vector<int>> *a2a_chunk_rem_size;
    subdemand_hash_t hash;
    SketchSubDemand(int layer_id, int group_id, intra_group_sched *sched, int split_chunks, double balance_ratio, uint64_t chunk_size_B , std::set<int> used_layers);
};

class Algorithm {
private:
    std::vector<Collective*> splitCollectives();
    std::vector<sched_t> searchEcGpus(Collective* collective);
    std::vector<sched_t> CustomizeSketch();
    void generate_layer_matrix(std::shared_ptr<Simulation::Topology> topo, std::vector<std::vector<int>>& layer_matrix, std::vector<std::vector<int>>& group_matrix);
    std::vector<std::shared_ptr<full_sched_graph_t>> expandSketches(std::vector<std::shared_ptr<sched_graph_t>> &scheds);
    std::vector<std::shared_ptr<combined_sketch_t>> combineSketches(std::vector<std::shared_ptr<full_sched_graph_t>> &sketches, bool a2a);
    // Chunk size for the single sketch
    std::vector<std::shared_ptr<solved_schedule_t>> sketchesSolve(std::vector<std::shared_ptr<sketch_solve_workload_t>> wrs, bool a2a);

    // Total chunk size split to all the sketches
    std::vector<std::shared_ptr<generated_algorithm_t>> combinedSketchesSolve(std::vector<std::shared_ptr<combined_sketch_t>> sketches, double balance_ratio, uint64_t chunk_size_B, int split_chunks, bool a2a, double &comb_sim_time);

public:
    nlohmann::json config;
    prune_policy_t prune;
    Collective* root_coll;
    std::shared_ptr<Simulation::PerfModel> perf_model;
    std::shared_ptr<Simulation::PerfModel> perf_model_pool[MAX_THREAD_NUM];
    std::shared_ptr<Simulation::Topology> topo_pool[MAX_THREAD_NUM];
    std::shared_ptr<Solver::AlgoSolverBase> solver;

    Algorithm(nlohmann::json& config, std::shared_ptr<Solver::AlgoSolverBase> solver);
    ~Algorithm();
    void generate();
};

std::vector<sched_t> search_sched_layers(prune_policy_t& p, std::shared_ptr<Simulation::Topology> topo, const sched_t& input);

std::vector<sched_t> search_sched_groups(prune_policy_t& policy, std::vector<sched_t>& inputs);

std::vector<sched_t> search_sched_steps(prune_policy_t& policy, std::vector<sched_t>& inputs);

struct DemandSolveReport {
    double iso_time_ms, solver_time_ms;
    int ndemands = 0, nsolves = 0;
};
// returns solved result, need to be mapped if demand_map_t is not empty
