// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/json.hpp"
#include "PerfModel/Topology.h"
#include "PerfModel/Type.h"
#include "Solver/common.h"
#include <memory>
#include <vector>
#include <cassert>

namespace Simulation {
    class SimConfig {
    public:
        std::shared_ptr<Solver::EventSolution> sol = nullptr;
        bool PXN = false;
    };
    
    class PerfModel {
    private:
        std::shared_ptr<Topology> topo;
        uint num_runs = 0;
        void initAnalysis();
        void resetAnalysis();

    public:
        PerfModel(std::shared_ptr<Topology> topo): topo(topo) {
          initAnalysis();
        }

        std::shared_ptr<Solver::SimResult> runAnalysis(const SimConfig &config);

        std::shared_ptr<Topology> getTopo() { return topo; };

        int num_chunks = 0;
    };

    // std::shared_ptr<Topology> topologyFromNetwork(Network *network);
    std::shared_ptr<Topology> constructTopology(nlohmann::json& config);
} // namespace Simulation
