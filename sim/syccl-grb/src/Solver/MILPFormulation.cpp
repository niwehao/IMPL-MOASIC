// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Solver/MIPSolver.h"
#include "PerfModel/Topology.h"
#include "PerfModel/Device.h"
#include "PerfModel/Link.h"
#include "Solver/Vars.hpp"
#include "Solver/common.h"
#include "Solver/ProcessTopology.h"
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

namespace Solver {

static void addBufferVar(std::shared_ptr<GRBModel> model,
                    int node, NodeInfo &info, const SourceCfg &sources, int nepoches,
                    BufferVars& buffer_vars, const MILPDemand &demand) {
  for (auto src_c : sources) {
    // only add buffer for src and dst in the demand
    if (info.type == NODE_GPU && !demand.at(src_c).count(node)) {
      continue;
    }
    for (int e = 0; e <= nepoches; ++e) {
      auto name = SolverVars::Buffer(src_c, node, e);
      // actually binary in MILP formulation
      auto v = model->addVar(0.0, 1, 0.0, GRB_BINARY, name);
      LOOOG(SOLVER, TRACE, "Adding Buffer: %s", name.c_str());
      buffer_vars[{src_c, node, e}] = v;
    }
  }
}

static void addIntegerForwardVar(std::shared_ptr<GRBModel> model,
                     ModeledLink &link, const SourceCfg &sources, int nepoches,
                     ForwardVars& forward_vars, int ub, const std::map<SrcChunk, int> &demand_sum, ForwardVarsUB &ubs, const MILPDemand &demand) {
  for (auto src_c : sources) {
    auto [src, dst, idx] = link.id;
    // Assumption: do not use nodes that are not in the demand
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
      int occ = link.info.capacity_chunk_double >= 1 ? link.info.occupy_e : 1;
      next_e = std::max(next_e, e + occ - 1);
      if (next_e > nepoches) {
        continue;
      }
      auto name = SolverVars::SendAtEpoch(src_c, link.id, e);
      int nub = std::min(ub, demand_sum.at(src_c));
      assert(nub >= 1);
      GRBVar v = model->addVar(0.0, nub, 0.0, GRB_INTEGER, name);
      LOOOG(SOLVER, TRACE, "Adding Forward: %s", name.c_str());
      forward_vars[{src_c, link.id, e}] = v;
      ubs[{src_c, link.id, e}] = nub;
    }
  }
}

static void addReceivedVar(std::shared_ptr<GRBModel> model,
                      int node, const SourceCfg &sources, int nepoches,
                      ReceivedVars& received_vars, const MILPDemand &demand) {
  for (auto src_c : sources) {
    if (!demand.at(src_c).count(node)) {
      continue;
    }
    if (demand.at(src_c).at(node) != 1) {
      continue;
    }
    for (int e = 0; e < nepoches; ++e) {
      auto name = SolverVars::Received(src_c, node, e);
      auto v = model->addVar(0.0, 1, 0.0, GRB_BINARY, name);
      LOOOG(SOLVER, TRACE, "Adding Received: %s", name.c_str());
      received_vars[{src_c, node, e}] = v;
    }
  }
}

static inline ModeledLink &getMLink(std::map<int, std::map<int, std::vector<ModeledLink>>> &link_info, LinkInfoId id) {
  auto [src, dst, idx] = id;
  return link_info[src][dst][idx];
}

MILPVarsMap MIPSolver::constructMILP(std::shared_ptr<GRBModel> model, std::shared_ptr<TopoInfo> topoinfo, const MILPDemand &demand, const SourceCfg &sources, int nepoches, ChunkSize chunk_size_B, double epoch_duration_ns) {
  LOOOG(SOLVER, TRACE, "Initializing MILP...");
  bool use_hyperedge = topoinfo->config.sw_hyperedge || topoinfo->config.nic_hyperedge;
  EpochModel epoch_model = config.epoch_model;
  // init all variables
  LOOOG(SOLVER, TRACE, "Creating MILP variables...");
  MILPVarsMap V;

  for (auto src_c : sources) {
    LOOOG(SOLVER, TRACE, "Source Chunk: %s", SolverVars::Chunk(src_c).c_str());
  }

  std::map<SrcChunk, int> demand_sum;
  for (auto [src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      demand_sum[src_c] += (v == 1);
      LOOOG(SOLVER, TRACE, "Demand: %s->%s: %d", SolverVars::Chunk(src_c).c_str(), SolverVars::Node(dst).c_str(), v);
    }
  }
  
  topoinfo->updateAllLinks(epoch_duration_ns, chunk_size_B, epoch_model);
  auto node_info = topoinfo->node_info;
  auto link_info = topoinfo->link_info;
  LOOOG(SOLVER, INFO, "Nepoches: %d, Chunk size: %.2lf KB, Epoch duration: %.2lf ns", nepoches, chunk_size_B / 1024.0, epoch_duration_ns);

  // Add Nodes
  for (auto &[i, info] : node_info) {
    LOOOG(SOLVER, TRACE, "Adding Node: %s", topoinfo->dev_names[i].c_str());
    if (info.sw_hyperedge && use_hyperedge) {
      // switch is replaced by hyperedges, no need for node
      continue;
    }
    if (info.buffer) {
      addBufferVar(model, i, info, sources, nepoches, V.B, demand);
    }
    if (info.recv) {
      addReceivedVar(model, i, sources, nepoches, V.R, demand);
    }
  }
  std::map<std::pair<double, double>, std::vector<LinkInfoId>> spec_to_link;
  LOOOG(SOLVER, TRACE, "Adding Links...");
  // Add Links
  for (auto &[src, srcm] : link_info) {
    for (auto &[dst, links] : srcm) {
      for (auto &link : links) {
        auto info = link.info;
        if (!link.present(use_hyperedge)) {
          continue;
        }
        LOOOG(SOLVER, TRACE, "Adding Link: %s", link.toStr().c_str());
        auto alpha = info.alpha, beta = info.beta;
        auto [src, dst, idx] = link.id;
        LOOOG(SOLVER, TRACE, "%s->%s", topoinfo->dev_names[src].c_str(), topoinfo->dev_names[dst].c_str());
        if (!spec_to_link.count({alpha, beta})) {
          spec_to_link[{alpha, beta}] = {};
          // LOOOG(SOLVER, TRACE, "New Link Type: %s", link.toStr().c_str());
        }
        spec_to_link[{alpha, beta}].push_back(link.id);
        // For switches doesn't copy, we may send multiple chunks continuously in one epoch and route them to different destinations
        if (!node_info[dst].copy) {
          int ub = std::max(1.0, floorThres(info.capacity_chunk_double));
          addIntegerForwardVar(model, link, sources, nepoches, V.F, ub, demand_sum, V.FUB, demand);
        } else {
          addIntegerForwardVar(model, link, sources, nepoches, V.F, 1, demand_sum, V.FUB, demand);
        }
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
  } */

  // *** Capacity constraints ***
  LOOOG(SOLVER, TRACE, "Adding capacity constraints...");
  // ((i->j), e) -> (src_c, (i->j), e), forwarding for each link
  // for hyperedges, we instead use the sum of all the transformed hyperedge links
  std::map<LinkInfoId, std::map<int, std::vector<GRBVar>>> link_vars;
  // No non-existent Links in V (i.e. original links replaced by hyperedges)
  for (auto &[k, v] : V.F) {
    auto [src_c, linkid, e] = k;
    auto &link = getMLink(link_info, linkid);
    if (link.is_hyperedge) {
      assert(use_hyperedge);
      for (auto linkid_o : link.hyperedge_info.link_groups) {
        auto link_o = getMLink(link_info, linkid_o);
        link_vars[linkid_o][e].push_back(v);
      }
    } else {
      assert(!(use_hyperedge && link.replace_by_hyperedge));
      link_vars[linkid][e].push_back(v);
    }
  }
  bool MODEL_EXTRA_SENDS_BIG_BETA = config.model_extra_sends_big_beta;
  bool MODEL_EXTRA_SENDS_SMALL_BETA = config.model_extra_sends_small_beta;
  assert(!(MODEL_EXTRA_SENDS_BIG_BETA || MODEL_EXTRA_SENDS_SMALL_BETA));
  // Add constraints
  for (auto &[linkid, emap] : link_vars) {
    auto &link = getMLink(link_info, linkid);
    double cap = link.info.capacity_chunk_double;
    std::map<int, GRBLinExpr> epoch_sum;
    for (int e = 0; e < nepoches; ++e) {
      GRBLinExpr expr;
      for (auto v : emap[e]) {
        expr += v;
      }
      epoch_sum[e] = expr;
    }
    if (cap >= 1) {
      for (int e = 0; e < nepoches; ++e) {
        modelAddConstr(model, epoch_sum[e] <= floorThres(cap), "link_capacity_" + SolverVars::Link(linkid) + "@" + SolverVars::Epoch(e));
      }
    } else {
      int occupy_e = ceilThres(1 / cap);
      for (int e = occupy_e - 1; e < nepoches; ++e) {
        GRBLinExpr ext;
        int l = e - occupy_e + 1;
        for (int i = l; i <= e; ++i) {
          ext += epoch_sum[i];
        }
        modelAddConstr(model, ext <= 1, "link_capacity_" + SolverVars::Link(linkid) + "@" + SolverVars::Epoch(e));
      }
    }
  }

  // *** Flow conservation and Buffer constraints ***
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

  // *** Buffer constraints ***
  LOOOG(SOLVER, TRACE, "Adding buffer constraints...");
  for (auto src_c : sources) {
    for (auto &[node, info] : node_info) {
      // model receive here for pipelined model
      if (epoch_model == EPOCH_MODEL_PIPELINE && info.recv && demand.at(src_c).count(node) && demand.at(src_c).at(node) == 1) {
        assert(V.R.count({src_c, node, 0}));
        auto first_recv = V.R[{src_c, node, 0}];
        modelAddConstr(model, first_recv == 0, "first_recv_constraint_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node));
        for (int e = 1; e < nepoches; ++e) {
          GRBLinExpr lhs, rhs;
          assert(V.R.count({src_c, node, e}));
          lhs += V.R[{src_c, node, e}];
          rhs += V.R[{src_c, node, e - 1}];
          for (auto [src, vm] : node_in_vars[{src_c, node}]) {
            for (auto [linkid, vm_e] : vm) {
              auto &link = getMLink(link_info, linkid);
              // assume switches have no buffers
              assert(link.type != LINK_GPU_TO_SWITCH && link.type != LINK_SWITCH_TO_SWITCH);
              // use full for pipelined recv
              int prev_e = link.get_prev_epoch_full(e);
              if (vm_e.count(prev_e)) {
                rhs += vm_e[prev_e];
              }
            }
          }
          modelAddConstr(model, lhs == rhs, "receive_constraint_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(e));
        }
      }
      if (V.B.count({src_c, node, 0})) {
        for (int e = 1; e <= nepoches; ++e) {
          GRBLinExpr lhs, rhs;
          assert(V.B.count({src_c, node, e}));
          lhs += V.B[{src_c, node, e}];
          rhs += V.B[{src_c, node, e - 1}];
          for (auto [src, vm] : node_in_vars[{src_c, node}]) {
            for (auto [linkid, vm_e] : vm) {
              auto &link = getMLink(link_info, linkid);
              // assume switches have no buffers
              assert(link.type != LINK_GPU_TO_SWITCH && link.type != LINK_SWITCH_TO_SWITCH);
              // possible to pipeline here
              int prev_e = link.get_prev_epoch(e);
              if (vm_e.count(prev_e)) {
                rhs += vm_e[prev_e];
              }
            }
          }
          modelAddConstr(model, lhs == rhs, "buffer_constraint_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(e));
        }
      }
    }
  }

  // Flow constraints
  LOOOG(SOLVER, TRACE, "Adding flow conservation constraints...");
  for (auto src_c : sources) {
    for (auto &[node, info] : node_info) {
      if (info.type == NODE_GPU && !V.B.count({src_c, node, 0})) {
        continue;
      }
      for (int e = 0; e < nepoches; ++e) {
        GRBLinExpr lhs;
        std::vector<std::pair<GRBVar, int>> rhs_vars;
        // lhs: received + buffer
        if (info.buffer) {
          assert(info.copy);
          // with buffers, received part is accounted in buffer constraints
          assert(V.B.count({src_c, node, e}));
          lhs += V.B[{src_c, node, e}];
        } else {
          // switches
          for (auto &[src, vm] : node_in_vars[{src_c, node}]) {
            for (auto &[linkid, vm_e] : vm) {
              auto &link = getMLink(link_info, linkid);
              // switch latency is accounted in the outgoing link
              int prev_e = link.get_prev_epoch(e);
              if (vm_e.count(prev_e)) {
                lhs += vm_e[prev_e];
              }
            }
          }
        }
        // rhs: sent out
        for (auto &[dst, vm] : node_out_vars[{src_c, node}]) {
          for (auto &[linkid, vm_e] : vm) {
            if (vm_e.count(e)) {
              assert(V.FUB.count({src_c, linkid, e}));
              int ub = V.FUB[{src_c, linkid, e}];
              rhs_vars.emplace_back(vm_e[e], ub);
            }
          }
        }
        if (rhs_vars.empty()) {
          continue;
        }
        if (info.copy) {
          assert(info.buffer);
          // maximum of sent out, multiple sends is possible as long as capacity constraint is respected
          int idx = 0;
          // max(rhs_vars) < lhs
          for (auto [var, ub] : rhs_vars) {
            // if received isn't zero(for a copy-enabled node, it should be binary to avoid waste of sends), then the node can send up to ub
            // ub should be reletivly small so that floating point errors(i.e. binary 0/1 is actually solved as double) won't affect the solution
            modelAddConstr(model, ub * lhs >= var, "flow_conservation_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(e) + "_" + std::to_string(idx++));
          }
        } else {
          // no copy, sum of sent out
          GRBLinExpr rhs;
          for (auto [var, _] : rhs_vars) {
            rhs += var;
          }
          // using >= is faster than ==
          modelAddConstr(model, lhs >= rhs, "flow_conservation_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(e));
        }
      }
    }
  }

  // *** initialization constraints ***
  LOOOG(SOLVER, TRACE, "Adding initialization constraints...");
  std::map<SrcChunk, int> tot_demand;
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      tot_demand[src_c] = std::max(tot_demand[src_c], v);
    }
  }
  // Add constraints for init -> 0
  for (auto src_c : sources) {
    for (auto &[node, info] : node_info) {
      if (!info.buffer || !V.B.count({src_c, node, 0})) {
        continue;
      }
      int init = (demand.at(src_c).count(node) && demand.at(src_c).at(node) == -1) ? tot_demand[src_c] : 0;
      if ((demand.at(src_c).count(node) && demand.at(src_c).at(node) == -1)) {
        assert(tot_demand[src_c] == 1);
      }
      auto b0 = V.B[{src_c, node, 0}];
      LOOOG(SOLVER, TRACE, "Adding init constraint: %s@%s@%s = %d", SolverVars::Chunk(src_c).c_str(), SolverVars::Node(node).c_str(), SolverVars::Epoch(0).c_str(), init);
      modelAddConstr(model, b0 == init, "init_constraint_" + SolverVars::Chunk(src_c) + "@" + SolverVars::Node(node) + "@" + SolverVars::Epoch(0));
    }
  }

  // *** Destination constraints ***
  LOOOG(SOLVER, TRACE, "Adding destination constraints...");
  for (auto &[src_c, dmap] : demand) {
    for (auto [dst, v] : dmap) {
      if (v != 1) {
        continue;
      }
      assert(node_info[dst].recv && node_info[dst].buffer);
      assert(V.R.count({src_c, dst, nepoches - 1}) && V.B.count({src_c, dst, nepoches}));
      auto final_recv = V.R[{src_c, dst, nepoches - 1}];
      modelAddConstr(model, final_recv == v, "Demand_constraint_" + SolverVars::Chunk(src_c) + "=>" + std::to_string(dst));
      if (epoch_model == EPOCH_MODEL_PIPELINE) {
        // PIPELINED RECEIVE is modeled at buffer constraints
        continue;
      }
      for (int e = 0; e < nepoches; ++e) {
        // A chunk is present in the buffer when it first arrives, but may not has fully arrived at the destination GPU, while it should only be received after it's full arrival.
        // However, considering uniform link chunk size, it's fine to just assume they re the same?
        // Besides, way more constraints are required since we have to calculate R from F in such cases.
        assert(V.B.count({src_c, dst, e + 1}) && V.R.count({src_c, dst, e}));
        auto recv = V.R[{src_c, dst, e}];
        auto buf = V.B[{src_c, dst, e + 1}];
        modelAddConstr(model, recv == buf, "destination_constraint_" + SolverVars::Chunk(src_c) + "=>" + std::to_string(dst) + "@" + SolverVars::Epoch(e));
      }
    }
  }

  // *** Objective ***
  LOOOG(SOLVER, TRACE, "Setting objective...");
  GRBLinExpr obj;
  for (int e = 0; e < nepoches; ++e) {
    for (auto &[src_c, dmap] : demand) {
      for (auto [dst, v] : dmap) {
        if (v != 1) continue;
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