// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Solver/MIPSolver.h"
#include "PerfModel/Topology.h"
#include "PerfModel/Device.h"
#include "PerfModel/Link.h"
#include "Solver/Vars.hpp"
#include "Solver/common.h"
#include "Solver/ProcessTopology.h"
#include "Sketch/common.h"
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include <cassert>
#include <chrono>
#include <memory>
#include <tuple>
#include <vector>

using namespace Solver;

std::pair<bool, MILPVarsMap> MIPSolver::solveMILPSAT(std::shared_ptr<TopoInfo> topoinfo, const MILPConfig &milp_config, int nepoches, double epoch_duration_ns) {
  auto demand = milp_config.demand;
  auto sources = milp_config.sources;
  ChunkSize chunk_size_B = config.chunk_size_B;
  assert(chunk_size_B != 0);
  setSatMode();
  auto V = constructMILP(model, topoinfo, demand, sources, nepoches, chunk_size_B, epoch_duration_ns);
  model->optimize();
  auto s = model->get(GRB_IntAttr_Status);
  LOOOG(SOLVER, TRACE, "Status: %d", s);
  bool feasible = (s == GRB_OPTIMAL || s == GRB_SOLUTION_LIMIT || s == GRB_SUBOPTIMAL);
  return {feasible, V};
}

// WARNING: Gurobi returns double values within a tolerance range, requiring manual rounding(and checking manually if the solution is feasible)
std::shared_ptr<MILPSolution> extractMILPSolution(MILPVarsMap V) {
  LOOOG(SOLVER, TRACE, "Extracting solution...");
  // *** Processing Solution ***
  // (Chunk, epoch) -> sends
  auto ret = std::make_shared<MILPSolution>();
  auto &sol = ret->sol;
  for (auto [_, f] : V.F) {
    auto name = f.get(GRB_StringAttr_VarName);
    double val = f.get(GRB_DoubleAttr_X);
    int val_r = std::round(val);
    if (val_r > 0) {
      // LOOOG(SOLVER, TRACE, "%s=%d", name.c_str(), val_r);
      auto [src_c, link, e] = _;
      for (int i = 0; i < val_r; ++i) {
        sol[src_c][e].push_back(link);
      }
    }
  }
  for (auto [_, c] : V.B) {
    auto name = c.get(GRB_StringAttr_VarName);
    double val = c.get(GRB_DoubleAttr_X);
    int val_r = std::round(val);
    if (val_r > 0) {
      // LOOOG(SOLVER, TRACE, "%s=%d", name.c_str(), val_r);
    }
  }
  return ret;
}

std::shared_ptr<MILPSolution> MIPSolver::_solveMILP(std::shared_ptr<TopoInfo> topoinfo, const MILPConfig &milp_config) {
  auto demand = milp_config.demand;
  auto sources = milp_config.sources;
  auto algo = milp_config.algo;
  ChunkSize chunk_size_B = config.chunk_size_B;
  assert(chunk_size_B != 0);

  auto epoch_time_strat = config.epoch_time_strat;
  EpochModel epoch_model = config.epoch_model;
  double balance_ratio = config.balance_ratio;
  double alpha_epoch_duration_ratio_max = config.alpha_epoch_duration_ratio_max;
  double epoch_duration_ns = topoinfo->getEpochTime(chunk_size_B, epoch_time_strat, balance_ratio, alpha_epoch_duration_ratio_max, epoch_model);

  int est_min_e = std::max(6, topoinfo->getMaxHopDist(true) * 2);
  int mhop = topoinfo->getMaxHopDist(true);
  double max_time = topoinfo->getSlowestChunkTime(chunk_size_B, mhop);
  int pair_count = 0;
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      pair_count += (v == 1);
    }
  }
  double est_time = max_time * est_min_e * pair_count / 2.0;

  double l = 0, r = est_time * config.trial_time_factor;
  int attempts = 0;
  double final_time = r;
  if (algo == SOLVE_NORMAL) {
    auto epoch_probe_start = std::chrono::high_resolution_clock::now();
    while (l < r) {
      double mid_t = (l + r) / 2;
      double solve_epoch_t = mid_t / est_min_e;
      LOOOG(SOLVER, TRACE, "***** Trying feasibility of %.6lf ns with %d epoches and epoch time %.6lf ns *****", mid_t, est_min_e, solve_epoch_t);
      if (solve_epoch_t < epoch_duration_ns && final_time != est_time * config.trial_time_factor) {
        break;
      }
      newModel();
      model->set(GRB_DoubleParam_TimeLimit, 60 * 10);
      auto [sat, SAT_V] = solveMILPSAT(topoinfo, milp_config, est_min_e, solve_epoch_t);
      if (sat) {
        int sat_e = findMILPSatEpoch(demand, SAT_V);
        double sat_t = (sat_e + 1) * solve_epoch_t;
        LOOOG(SOLVER, TRACE, "SAT in %d epoches, %.6lf ns", sat_e + 1, sat_t);
        r = sat_t;
        final_time = std::min(final_time, sat_t);
      } else {
        l = mid_t;
      }
      if (++attempts > 10) {
        break;
      }
    }
    auto epoch_probe_end = std::chrono::high_resolution_clock::now();

    int nepoches = ceil(final_time / epoch_duration_ns);
    LOOOG(SOLVER, INFO, "Final epoches: %d, Final time: %.6lf ns", nepoches, final_time);
    newModel();

    auto construct_start = std::chrono::high_resolution_clock::now();
    auto V = constructMILP(model, topoinfo, demand, sources, nepoches, chunk_size_B, epoch_duration_ns);
    auto construct_end = std::chrono::high_resolution_clock::now();

    // *** Solving ***
    int ntries = 1;
    LOOOG(SOLVER, TRACE, "Optimizing MILP...");
    auto solve_start = std::chrono::high_resolution_clock::now();
    model->optimize();

    while (model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
      if (ntries >= 5) {
        LOOOG(SOLVER, INFO, "Infeasible after 5 tries");
        assert(ntries < 5);
      }
      nepoches = ceil(nepoches * 1.2);
      ntries++;
      LOOOG(SOLVER, INFO, "Infeasible, retrying with %d epoches, Trial %d", nepoches, ntries);
      newModel();
      V = constructMILP(model, topoinfo, demand, sources, nepoches, chunk_size_B, epoch_duration_ns);
      model->optimize();
    }

    if (model->get(GRB_IntAttr_Status)== GRB_TIME_LIMIT) {
      LOOOG(SOLVER, INFO, "Time limit reached");
      int sol_count = model->get(GRB_IntAttr_SolCount);
      LOOOG(SOLVER, INFO, "Solution count: %d", sol_count);
      if (sol_count == 0) {
        assert(false);
      }
    }

    auto solve_end = std::chrono::high_resolution_clock::now();
    int sat_e = findMILPSatEpoch(demand, V);
    LOOOG(SOLVER, INFO, "SAT in %d epoches, time: %.6lf ns", (sat_e + 1), (sat_e + 1) * epoch_duration_ns);
    
    auto ret = extractMILPSolution(V);
    ret->epoch_duration_ns = epoch_duration_ns;
    ret->nepoches = sat_e + 1;
    ret->finish_time_ns = (sat_e + 1) * epoch_duration_ns;
    
    ret->epoch_probe_time = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_probe_end - epoch_probe_start).count() / 1e9;
    ret->construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(construct_end - construct_start).count() / 1e9;
    ret->solve_time = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count() / 1e9;

    ret->nsolves = ntries;
    return ret;
  } else if (algo == SOLVE_BY_SAT) {
    int l = est_min_e / 2, r = est_min_e * 5;
    MILPVarsMap V_sat;
    while (l < r) {
      int m = (l + r) / 2;
      LOOOG(SOLVER, INFO, "Trying %d epoches...", m);
      newModel();
      auto [sat, V] = solveMILPSAT(topoinfo, milp_config, m, epoch_duration_ns);
      if (sat) {
        LOOOG(SOLVER, INFO, "Solved with %d epoches", m);
        r = m, V_sat = V;
        auto ret = extractMILPSolution(V_sat);
      } else {
        l = m + 1;
      }
    }
    auto ret = extractMILPSolution(V_sat);
    return ret;
  } else {
    assert(false);
  }
}

int MIPSolver::findMILPSatEpoch(const MILPDemand &demand, const MILPVarsMap& V) {
  std::map<std::pair<SrcChunk, int>, int> sat_epoch;
  for (auto [_, var] : V.B) {
    double val = var.get(GRB_DoubleAttr_X);
    int val_r = std::round(val);
    if (val_r > 0) {
      auto name = var.get(GRB_StringAttr_VarName);
      // LOOOG(SOLVER, TRACE, "%s=%d", name.c_str(), val_r);
    }
  }
  for (auto [_, var] : V.F) {
    double val = var.get(GRB_DoubleAttr_X);
    int val_r = std::round(val);
    if (val_r > 0) {
      auto name = var.get(GRB_StringAttr_VarName);
      // LOOOG(SOLVER, TRACE, "%s=%d", name.c_str(), val_r);
    }
  }
  for (auto [_, var] : V.R) {
    double val = var.get(GRB_DoubleAttr_X);
    int val_r = std::round(val);
    if (val_r > 0) {
      auto name = var.get(GRB_StringAttr_VarName);
      // LOOOG(SOLVER, TRACE, "%s=%d", name.c_str(), val_r);
      auto [src_c, node, e] = _;
      if (!sat_epoch.count({src_c, node})) {
        sat_epoch[{src_c, node}] = e;
      } else {
        sat_epoch[{src_c, node}] = std::min(sat_epoch[{src_c, node}], e);
      }
    }
  }
  int sat_e = -1;
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      if (v <= 0) continue;
      if (!sat_epoch.count({src_c, dst})) {
        return -1;
      }
      sat_e = std::max(sat_e, sat_epoch[{src_c, dst}]);
    }
  }
  return sat_e;
}