// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Solver/MIPSolver.h"
#include "PerfModel/Device.h"
#include "PerfModel/Topology.h"
#include "Solver/Vars.hpp"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include <memory>
#include <tuple>
#include <vector>

namespace Solver {

static void addLPBufferVar(std::shared_ptr<GRBModel> model,
                    int node, NodeInfo &info, const SourceCfg &sources, int nepoches, std::map<SrcChunk, double> &tot_demand,
                    BufferVars& buffer_vars, const LPDemand &demand) {
  for (auto src_c : sources) {
    // only add buffer for src and dst in the demand
    if (info.type == NODE_GPU && !demand.at(src_c).count(node)) {
      continue;
    }
    for (int e = 0; e <= nepoches; ++e) {
      auto name = SolverVars::Buffer(src_c, node, e);
      // total of chunks for each src
      double ub = tot_demand[src_c];
      auto v = model->addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
      buffer_vars[{src_c, node, e}] = v;
    }
  }
}

static void addLPForwardVar(std::shared_ptr<GRBModel> model,
                     ModeledLink &link, const SourceCfg &sources, int nepoches, const std::map<SrcChunk, double> &tot_demand, 
                     ForwardVars& forward_vars, const LPDemand &demand) {
  for (auto src_c : sources) {
    auto [src, dst, idx] = link.id;
    if (demand.at(src_c).count(dst) && demand.at(src_c).at(dst) < 0) {
      continue;
    }
    if (dstisGPU(link.type)) {
      if (!demand.at(src_c).count(dst)) {
        continue;
      }
      if (demand.at(src_c).at(dst) == -1) {
        continue;
      }
    }
    if (srcisGPU(link.type)) {
      if (!demand.at(src_c).count(src)) {
        continue;
      }
    }
    for (int e = 0; e < nepoches; ++e) {
      int prev_e = link.get_prev_epoch_full(e);
      int next_e = e + (e - prev_e);
      if (next_e > nepoches) {
        continue;
      }
      auto name = SolverVars::SendAtEpoch(src_c, link.id, e);
      double ub = tot_demand.at(src_c);
      auto v = model->addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
      forward_vars[{src_c, link.id, e}] = v;
    }
  }
}

static void addLPConsumeVar(std::shared_ptr<GRBModel> model,
                     int node, const SourceCfg &sources, int nepoches,
                     ConsumeVars& consume_vars, const LPDemand &demand) {
  for (auto src_c : sources) {
    if (!demand.at(src_c).count(node)) {
      continue;
    }
    if (demand.at(src_c).at(node) < 0) {
      continue;
    }
    for (int e = 0; e <= nepoches; ++e) {
      auto name = SolverVars::Consume(src_c, node, e);
      auto ub = demand.at(src_c).at(node);
      auto v = model->addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
      consume_vars[{src_c, node, e}] = v;
    }
  }
}

static void addLPReceivedVar(std::shared_ptr<GRBModel> model,
                      int node, const SourceCfg &sources, int nepoches,
                      ReceivedVars& received_vars, const LPDemand &demand) {
  for (auto src_c : sources) {
    if (!demand.at(src_c).count(node)) {
      continue;
    }
    if (demand.at(src_c).at(node) < 0) {
      continue;
    }
    int ub = demand.at(src_c).at(node);
    for (int e = 0; e <= nepoches; ++e) {
      auto name = SolverVars::Received(src_c, node, e);
      auto v = model->addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
      received_vars[{src_c, node, e}] = v;
    }
  }
}

static inline ModeledLink &getMLink(std::map<int, std::map<int, std::vector<ModeledLink>>> &link_info, LinkInfoId id) {
  auto [src, dst, idx] = id;
  return link_info[src][dst][idx];
}

LPVarsMap MIPSolver::constructLP(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topoinfo, const LPDemand &demand, const SourceCfg &sources, int nepoches, ChunkSize chunk_size_B, double epoch_duration_ns) {
  LOOOG(SOLVER, TRACE, "Initializing LP...");
  bool use_hyperedge = topoinfo->config.sw_hyperedge || topoinfo->config.nic_hyperedge;
  EpochModel epoch_model = config.epoch_model;

  // init all variables
  LOOOG(SOLVER, TRACE, "Creating LP variables...");
  LPVarsMap V;

  std::map<SrcChunk, double> tot_demand, tot_supply;
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      if (v > 0) {
        tot_demand[src_c] += v;
      } else {
        tot_supply[src_c] += v;
      }
    }
  }
  for (auto &[src_c, v] : tot_supply) {
    assert(fabs(-tot_supply[src_c] - tot_demand[src_c]) < 1e-6);
  }

  topoinfo->updateAllLinks(epoch_duration_ns, chunk_size_B, epoch_model);
  auto node_info = topoinfo->node_info;
  auto link_info = topoinfo->link_info;
  LOOOG(SOLVER, TRACE, "Nepoches: %d, Chunk size: %.2lf KB, Epoch duration: %.2lf ns", nepoches, chunk_size_B / 1024.0, epoch_duration_ns);

  for (auto &[i, info] : node_info) {
    if (info.sw_hyperedge && use_hyperedge) {
      // switch is replaced by hyperedges, no need for node
      continue;
    }
    if (info.buffer) {
      addLPBufferVar(model, i, info, sources, nepoches, tot_demand, V.B, demand);
    }
    if (info.recv) {
      addLPConsumeVar(model, i, sources, nepoches, V.C, demand);
      addLPReceivedVar(model, i, sources, nepoches, V.R, demand);
    }
  }
  std::map<std::pair<double, double>, std::vector<LinkInfoId>> spec_to_link;
  LOOOG(SOLVER, TRACE, "Adding link variables...");
  for (auto &[src, srcm] : link_info) {
    for (auto &[dst, links] : srcm) {
      for (auto &link : links) {
        if (!link.present(use_hyperedge)) {
          continue;
        }
        auto info = link.info;
        auto alpha = info.alpha, beta = info.beta;
        // auto [src, dst, idx] = link.id;
        // LOOOG(SOLVER, INFO, "Link %s->%s", topoinfo->dev_names[src].c_str(), topoinfo->dev_names[dst].c_str());
        if (!spec_to_link.count({alpha, beta})) {
          spec_to_link[{alpha, beta}] = {};
          // LOOOG(SOLVER, INFO, "New Link Type: %s", link.toStr().c_str());
        }
        spec_to_link[{alpha, beta}].push_back(link.id);
        addLPForwardVar(model, link, sources, nepoches, tot_demand, V.F, demand);
      }
    }
  }
  /*
  for (auto &[spec, links] : spec_to_link) {
    auto linkid = links[0];
    LOOOG(SOLVER, TRACE, "Link Type: %s", getMLink(link_info, linkid).toStr().c_str());
    std::string link_str;
    for (auto &[src, dst, idx] : links) {
      link_str += format("%s->%s, ", topoinfo->dev_names[src].c_str(), topoinfo->dev_names[dst].c_str());
    }
    LOOOG(SOLVER, TRACE, "Links: %s", link_str.c_str());
  }
  */

  // *** Capacity constraints ***
  LOOOG(SOLVER, TRACE, "Adding capacity constraints...");
  // ((i->j), e) -> (src_c, (i->j), e), forwarding for each link
  std::map<std::pair<LinkInfoId, int>, std::vector<GRBVar>> link_vars;
  for (auto &[k, v] : V.F) {
    auto &[src_c, linkid, e] = k;
    // LP handles capacity constraint via fractional chunks automatically
    auto &link = getMLink(link_info, linkid);
    if (link.is_hyperedge) {
      assert(use_hyperedge);
      for (auto linkid_o : link.hyperedge_info.link_groups) {
        link_vars[{linkid_o, e}].push_back(v);
      }
    } else {
      assert(!(use_hyperedge && link.replace_by_hyperedge));
      link_vars[{linkid, e}].push_back(v);
    }
  }
  // Add constraints
  for (auto &[cc, vars] : link_vars) {
    GRBLinExpr expr;
    for (auto v : vars) {
      expr += v;
    }
    auto [linkid, e] = cc;
    auto &link = getMLink(link_info, linkid);
    auto cap_e = link.info.capacity_chunk_double;
    modelAddConstr(model, expr <= cap_e, "link_capacity_" + SolverVars::Link(linkid) + "@" + SolverVars::Epoch(e));
  }

  // *** Flow conservation(and buffer) constraints ***
  LOOOG(SOLVER, TRACE, "Adding flow conservation constraints...");
  // (src_c, dst) -> (src -> (link -> e -> var)), receiving end for each node
  std::map<std::tuple<SrcChunk, int>,
    std::map<int,
      std::map<LinkInfoId,
        std::map<int,
          GRBVar
  >>>> node_in_vars;
  // (src_c, src) -> (dst -> (link -> e -> var)), sending end for each node
  std::map<std::tuple<SrcChunk, int>,
    std::map<int,
      std::map<LinkInfoId,
        std::map<int,
          GRBVar
  >>>> node_out_vars;
  for (auto &[k, v] : V.F) {
    auto [src_c, linkid, e] = k;
    auto [src, dst, idx] = linkid;
    node_in_vars[{src_c, dst}][src][linkid][e] = v;
    node_out_vars[{src_c, src}][dst][linkid][e] = v;
  }
  // Add constraints
  for (auto &src_c : sources) {
    for (auto &[node, info] : node_info) {
      for (int e = 0; e < nepoches; ++e) {
        GRBLinExpr lhs, rhs;
        if (V.B.count({src_c, node, 0})) {
          assert(V.B.count({src_c, node, e}) && V.B.count({src_c, node, e + 1}));
          // with buffers
          lhs += V.B[{src_c, node, e}];
          rhs += V.B[{src_c, node, e + 1}];
          if (V.C.count({src_c, node, e})) {
            rhs += V.C[{src_c, node, e}];
          }
        }
        for (auto &[src, vm] : node_in_vars[{src_c, node}]) {
          for (auto &[linkid, vm_e] : vm) {
            auto &link = getMLink(link_info, linkid);
            int prev_e = link.get_prev_epoch_full(e);
            if (vm_e.count(prev_e)) {
              lhs += vm_e[prev_e];
            }
          }
        }
        for (auto &[dst, vm] : node_out_vars[{src_c, node}]) {
          for (auto &[linkid, vm_e] : vm) {
            if (vm_e.count(e)) {
              rhs += vm_e[e];
            }
          }
        }
        modelAddConstr(model, lhs == rhs, "flow_conservation_src_chunk_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(e));
      }
    }
  }

  // *** initialization and termination constraints ***
  LOOOG(SOLVER, TRACE, "Adding initialization constraints...");
  // Add constraints for init -> 0
  for (auto src_c : sources) {
    auto [id, pos] = src_c;
    for (auto &[node, info] : node_info) {
      if (!info.buffer || !V.B.count({src_c, node, 0})) {
        continue;
      }
      double init = demand.at(src_c).count(node) ?
                    demand.at(src_c).at(node) < 0 ? tot_demand[src_c] : 0
                    : 0;
      auto b0 = V.B[{src_c, node, 0}];
      auto fs = node_out_vars[{src_c, node}];
      GRBLinExpr ep0 = b0;
      for (auto &[dst, vm] : fs) {
        for (auto &[linkid, vm_e] : vm) {
          ep0 += vm_e[0];
        }
      }
      modelAddConstr(model, ep0 == init, "init_constraint_src_chunk_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(0));
    }
  }
  // last epoch
  LOOOG(SOLVER, TRACE, "Adding termination constraints...");
  int last_e = nepoches;
  for (auto src_c : sources) {
    for (auto &[node, info] : node_info) {
      if (!info.recv || !demand.at(src_c).count(node) || demand.at(src_c).at(node) < 0) {
        continue;
      }
      if (!V.C.count({src_c, node, last_e})) {
        continue;
      }
      auto consume = V.C[{src_c, node, last_e}];
      GRBLinExpr recv = 0;
      for (auto &[src, vm] : node_in_vars[{src_c, node}]) {
        for (auto &[linkid, vm_e] : vm) {
          auto &link = getMLink(link_info, linkid);
          int prev_e = link.get_prev_epoch_full(last_e);
          if (vm_e.count(prev_e)) {
            recv += vm_e[prev_e];
          }
        }
      }
      modelAddConstr(model, consume == recv, "termination_constraint_src_chunk_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(last_e));
    }
  }

  // *** Destination constraints ***
  LOOOG(SOLVER, TRACE, "Adding destination constraints...");
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      if (v <= 0) {
        continue;
      }
      for (int e = 0; e <= nepoches; ++e) {
        GRBLinExpr consumed = V.C[{src_c, dst, e}];
        if (e > 0) {
          consumed += V.R[{src_c, dst, e - 1}];
        }
        auto recv = V.R[{src_c, dst, e}];
        modelAddConstr(model, recv == consumed, "destination_constraint_src_chunk_" + SolverVars::Chunk(src_c) + "=>" + std::to_string(dst) + "@" + SolverVars::Epoch(e));
      }
      auto final_recv = V.R[{src_c, dst, nepoches}];
      modelAddConstr(model, final_recv == v, "Demand_constraint_src_chunk_" + SolverVars::Chunk(src_c) + "=>" + std::to_string(dst));
    }
  }

  // *** Objective ***
  LOOOG(SOLVER, TRACE, "Setting objective...");
  GRBLinExpr obj;
  // TODO: different objective in TECCL source code?
  for (int e = 0; e <= nepoches; ++e) {
    for (auto &[src_c, dmap] : demand) {
      for (auto [dst, v] : dmap) {
        if (v <= 0) continue;
        assert(V.R.count({src_c, dst, e}));
        obj += V.R[{src_c, dst, e}] / (e + 1);
      }
    }
  }
  try {
    model->setObjective(obj, GRB_MAXIMIZE);
  } catch (GRBException& e) {
    handleGRBException(e);
  }
  return V;
}

} // namespace Solver