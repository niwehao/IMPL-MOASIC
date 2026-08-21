// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PerfModel/Type.h"
#include "Solver/ProcessTopology.h"
#include "Solver/common.h"
#include "common/json.hpp"
#include <memory>

namespace Solver {
  struct SolverConfig {
    ChunkSize chunk_size_B;
    int split_chunks;
    double alpha_threshold;
    double alpha_epoch_duration_ratio_max;
    double balance_ratio;
    double trial_time_factor;
    EpochTimeStrategy epoch_time_strat;
    EpochModel epoch_model;
    bool model_extra_sends_big_beta, model_extra_sends_small_beta;
    SolverConfig(nlohmann::json config);
  };

  class AlgoSolverBase {
  public:
    std::shared_ptr<Simulation::Topology> topo;
    SolverConfig config;
    AlgoSolverBase(std::shared_ptr<Simulation::Topology> topo, SolverConfig config);
    virtual ~AlgoSolverBase() = default;

    virtual std::shared_ptr<EventSolution> solveMILP(MILPConfig milp_config, TopoConfig topo_config) = 0;
    virtual std::shared_ptr<EventSolution> solveLP(LPConfig lp_config, TopoConfig topo_config) = 0;

    std::shared_ptr<EventSolution> extractMILPEventSolution(std::shared_ptr<MILPSolution> solution, std::shared_ptr<TopoInfo> topo_info, MILPDemand &demand, ChunkSize chunk_size_B);
    std::shared_ptr<EventSolution> extractLPEventSolution(std::shared_ptr<LPSolution> solution, std::shared_ptr<TopoInfo> topo_info, LPDemand &demand, ChunkSize chunk_size_B);
  };
} // namespace Solver