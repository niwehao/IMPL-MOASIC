// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Solver/MIPSolver.h"
#include "PerfModel/Type.h"
#include "PerfModel/Device.h"
#include "Solver/Vars.hpp"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "gurobi_c++.h"
#include <memory>

inline std::mutex cnt_mutex;
namespace Solver {

int MIPSolver::num_lp_solves = 0;
int MIPSolver::num_milp_solves = 0;
std::vector<int> MIPSolver::all_nsolves, MIPSolver::all_nepoches;
std::vector<double> MIPSolver::all_epoch_duration_ns, MIPSolver::all_finish_time_ns, MIPSolver::all_epoch_probe_time, MIPSolver::all_construct_time, MIPSolver::all_solve_time;

std::shared_ptr<MIPSolver> MIPSolver::clone() {
  auto ret = std::make_shared<MIPSolver>(topo, config, mip_params);
  return ret;
}

MIPSolver::MIPSolver(std::shared_ptr<Simulation::Topology> topo, nlohmann::json& config) : AlgoSolverBase(topo, SolverConfig(config)), mip_params(config["mip_config"]) {
  env = std::make_shared<GRBEnv>(true);
  env->set("LogFile", mip_params.log_file);
  env->set(GRB_IntParam_OutputFlag, 0);
  GRB_CHECK(env->start());
}

MIPSolver::MIPSolver(std::shared_ptr<Simulation::Topology> topo, SolverConfig& config, MIPParams& params) : AlgoSolverBase(topo, config), mip_params(params) {
  env = std::make_shared<GRBEnv>(true);
  env->set("LogFile", mip_params.log_file);
  env->set(GRB_IntParam_OutputFlag, 0);
  GRB_CHECK(env->start());
}

void MIPSolver::newModel() {
  auto params = mip_params;
  model = std::make_shared<GRBModel>(*env);
  model->set(GRB_DoubleParam_FeasibilityTol, params.feasibility_tol);
  model->set(GRB_DoubleParam_IntFeasTol, params.int_feas_tol);
  model->set(GRB_DoubleParam_OptimalityTol, params.optimality_tol);
  model->set(GRB_DoubleParam_MIPGap, params.mip_gap);
  model->set(GRB_DoubleParam_Heuristics, params.heuristics);
  model->set(GRB_IntParam_MIPFocus, params.mip_focus);
  model->set(GRB_IntParam_SolutionLimit, params.solution_limit);
  model->set(GRB_DoubleParam_TimeLimit, params.time_limit_h * 3600);
}

void MIPSolver::setSatMode() {
  model->set(GRB_IntParam_MIPFocus, 1);
  model->set(GRB_IntParam_SolutionLimit, 1);
}

void handleGRBException(GRBException& e) {
  std::cerr << "GRBSolver Error code = " << e.getErrorCode() << ", " << e.getMessage() << std::endl;
  exit(0);
}

void modelAddConstr(std::shared_ptr<GRBModel> model, const GRBTempConstr& tc, std::string name) {
  try {
    model->addConstr(tc, name);
  } catch (GRBException& e) {
    handleGRBException(e);
  }
}

std::shared_ptr<EventSolution> MIPSolver::solveLP(LPConfig lp_config, TopoConfig topo_config) {
  auto topo_info = std::make_shared<TopoInfo>(topo, topo_config);
  auto solution = _solveLP(topo_info, lp_config);
{
  std::lock_guard<std::mutex> lock(cnt_mutex);
  LOOOG(ALGO, INFO, "LPsolve num: %d", num_lp_solves);
  // LOOOG(SOLVER, INFO, "Solution metadata: %s", solution->metatoStr().c_str());
  num_lp_solves++;
  all_nepoches.push_back(solution->nepoches);
  all_nsolves.push_back(solution->nsolves);
  all_epoch_duration_ns.push_back(solution->epoch_duration_ns);
  all_finish_time_ns.push_back(solution->finish_time_ns);
  all_epoch_probe_time.push_back(solution->epoch_probe_time);
  all_construct_time.push_back(solution->construct_time);
  all_solve_time.push_back(solution->solve_time);
}
  return extractLPEventSolution(solution, topo_info, lp_config.demand, config.chunk_size_B);
}

std::shared_ptr<EventSolution> MIPSolver::solveMILP(MILPConfig milp_config, TopoConfig topo_config) {
  auto topo_info = std::make_shared<TopoInfo>(topo, topo_config);
  auto solution = _solveMILP(topo_info, milp_config);
  // record metadata
{
  std::lock_guard<std::mutex> lock(cnt_mutex);
  LOOOG(ALGO, INFO, "MILPsolve #%d Finished", num_milp_solves);
  // LOOOG(SOLVER, INFO, "Solution metadata: %s", solution->metatoStr().c_str());
  num_milp_solves++;
  all_nepoches.push_back(solution->nepoches);
  all_nsolves.push_back(solution->nsolves);
  all_epoch_duration_ns.push_back(solution->epoch_duration_ns);
  all_finish_time_ns.push_back(solution->finish_time_ns);
  all_epoch_probe_time.push_back(solution->epoch_probe_time);
  all_construct_time.push_back(solution->construct_time);
  all_solve_time.push_back(solution->solve_time);
}
  return extractMILPEventSolution(solution, topo_info, milp_config.demand, config.chunk_size_B);
}

void MIPSolver::setBalanceRatio(double ratio) {
  config.balance_ratio = ratio;
}

void MIPSolver::setChunkSize(ChunkSize chunk_size) {
  config.chunk_size_B = chunk_size;
}

}// namespace Solver