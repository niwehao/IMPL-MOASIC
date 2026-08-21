// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/json.hpp"
#include "Solver/AlgoSolver.h"
#include "Solver/common.h"
#include "Solver/ProcessTopology.h"
#include "gurobi_c++.h"
#include <memory>

#define GRB_CHECK(call)               \
  try {                               \
    call;                             \
  } catch (GRBException& e) {         \
    std::cerr << "Gurobi exception: " \
              << "Error code = " << e.getErrorCode() << ", " \
              << "Message = " << e.getMessage() << std::endl; \
    exit(0); \
  } catch (std::exception& e) {       \
    std::cerr << "Std exception: " << e.what() << std::endl; \
    exit(0); \
  } catch (...) {                     \
    std::cerr << "Unknown exception caught!" << std::endl; \
    exit(0); \
  }

namespace Solver {
  struct MIPParams {
    double feasibility_tol;
    double int_feas_tol;
    double optimality_tol;
    double mip_gap;
    double heuristics;
    int mip_focus;
    int solution_limit;
    double time_limit_h;
    int verbose_level;
    std::string log_file;
    MIPParams(nlohmann::json mip_config) {
      feasibility_tol = mip_config.value("feasibility_tol", 1e-4);
      int_feas_tol = mip_config.value("intfeas_tol", 1e-4);
      mip_gap = mip_config.value("mip_gap", 1e-4);
      optimality_tol = mip_config.value("optimality_tol", 1e-4);
      mip_focus = mip_config.value("mip_focus", 1);
      heuristics = mip_config.value("heuristics", 0.25);
      solution_limit = mip_config.value("solution_limit", 2000000);
      log_file = mip_config.value("log_file", "mip_solve.log");
      time_limit_h = mip_config.value("time_limit_h", 2.0);
      verbose_level = mip_config.value("verbose_level", 3);
    }
    MIPParams(): MIPParams(nlohmann::json()) {}
  };

  class MIPSolver: public AlgoSolverBase {
  private:
    std::shared_ptr<GRBEnv> env;
    std::shared_ptr<GRBModel> model;
    MIPParams mip_params;
  public:
    MIPSolver(std::shared_ptr<Simulation::Topology> topo, nlohmann::json& config);
    MIPSolver(std::shared_ptr<Simulation::Topology> topo, SolverConfig& config, MIPParams& params);
    ~MIPSolver() = default;

    void setBalanceRatio(double ratio);
    void setChunkSize(ChunkSize chunk_size);
    std::shared_ptr<MIPSolver> clone();

    std::shared_ptr<EventSolution> solveMILP(MILPConfig milp_config, TopoConfig topo_config);
    std::shared_ptr<EventSolution> solveLP(LPConfig lp_config, TopoConfig topo_config);

    bool testMILPEpochFeasible(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topo, const MILPConfig &milp_config, int ne, double epoch_duration_ns);
    std::shared_ptr<MILPSolution> _solveMILP(std::shared_ptr<TopoInfo> topoinfo, const MILPConfig &milp_config);
    std::pair<bool, MILPVarsMap> solveMILPSAT(std::shared_ptr<TopoInfo> topo, const MILPConfig &milp_config, int nepoches, double epoch_duration_ns);
    MILPVarsMap constructMILP(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topoinfo, const MILPDemand &demand, const SourceCfg &sources, int nepoches, ChunkSize chunk_size_B, double epoch_duration_ns);
    int findMILPSatEpoch(const MILPDemand &demand, const MILPVarsMap& V);

    bool testLPEpochFeasible(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topo, const LPConfig &lp_config, int ne, double epoch_duration_ns);
    std::shared_ptr<LPSolution> _solveLP(std::shared_ptr<TopoInfo> topo, const LPConfig &lp_config);
    std::pair<bool, LPVarsMap> solveLPSAT(std::shared_ptr<TopoInfo> topo, const LPConfig &lp_config, int nepoches, double epoch_duration_ns);
    LPVarsMap constructLP(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topoinfo, const LPDemand &demand, const SourceCfg &sources, int nepoches, ChunkSize chunk_size_B, double epoch_duration_ns);
    int findLPSatEpoch(const LPDemand &demand, const LPVarsMap& V);

    void newModel();
    void setSatMode();

    static int num_lp_solves, num_milp_solves;
    static std::vector<int> all_nepoches, all_nsolves;
    static std::vector<double> all_epoch_duration_ns, all_finish_time_ns, all_epoch_probe_time, all_construct_time, all_solve_time;
  };

  void handleGRBException(GRBException& e);
  void modelAddConstr(std::shared_ptr<GRBModel> model, const GRBTempConstr& tc, std::string name);

} // namespace Solver