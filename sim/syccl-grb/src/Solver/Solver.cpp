// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Solver/common.h"
#include "Solver/AlgoSolver.h"
#include "Solver/ProcessTopology.h"
#include "Solver/Vars.hpp"
#include "common/log.h"
#include "PerfModel/Topology.h"
#include <cassert>
#include <memory>
#include <string>

using namespace Solver;

AlgoSolverBase::AlgoSolverBase(std::shared_ptr<Simulation::Topology> topo, SolverConfig config): topo(topo), config(config) {
}

// Translate a solution into concrete sends, which rquires finding every parent send for receives that passes through a switch
std::shared_ptr<EventSolution> AlgoSolverBase::extractMILPEventSolution(std::shared_ptr<MILPSolution> solution, std::shared_ptr<TopoInfo> topo_info, MILPDemand &demand, ChunkSize chunk_size_B) {
  auto event_sol = std::make_shared<EventSolution>();
  auto sol = solution->sol;
  auto dev_names = topo_info->dev_names;
  auto node_info = topo_info->node_info;
  for (auto [src_c, epoches] : sol) {
    LOOOG(SOLVER, TRACE, "------ Solution for %s ------", SolverVars::Chunk(src_c).c_str());
    // Extract sends from solution (dev -> (epoch, src))
    std::map<int, std::vector<std::pair<int, int>>> sol_i;
    for (auto [e, links] : epoches) {
      LOOOG(SOLVER, TRACE, "%s@%s", SolverVars::Chunk(src_c).c_str(), SolverVars::Epoch(e).c_str());
      for (auto [src, dst, id] : links) {
        LOOOG(SOLVER, TRACE, "%s -> %s", dev_names[src].c_str(), dev_names[dst].c_str());
        sol_i[dst].push_back({e, src});
      }
    }
    for (auto [dev, ev] : sol_i) {
      std::sort(ev.begin(), ev.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
        return a.first < b.first;
      });
    }
    std::vector<int> dsts;
    for (auto[dev, amount] : demand[src_c]) {
      if (amount > 0 && !sol_i[dev].empty()) {
        dsts.push_back(dev);
      }
    }
    // find the latest epoches first to account for switch link matching
    std::sort(dsts.begin(), dsts.end(), [&sol_i](int a, int b) {
      return sol_i[a].begin()->first > sol_i[b].begin()->first;
    });
    for (auto dev : dsts) {
      // init with the earliest epoch
      int prev_e = 1e9, src = dev, dst = dev;
      do {
        bool f = false;
        assert(!sol_i[src].empty());
        for (auto t = sol_i[src].rbegin(); t != sol_i[src].rend(); ++t) {
          // src0 -> src
          auto [e, src0] = *t;
          LOOOG(SOLVER, DEBUG, "Link: %s -> %s: e = %d", dev_names[src0].c_str(), dev_names[src].c_str(), e);
          // Do not allow multiple links between two nodes here
          assert(topo_info->link_info[src0][src].size() == 1);
          auto link = topo_info->link_info[src0][src][0];
          int prev_prev_e = link.get_prev_epoch(prev_e);
          LOOOG(SOLVER, DEBUG, "Link: %s -> %s: prev_e = %d, prev_prev_e = %d", dev_names[src0].c_str(), dev_names[src].c_str(), prev_e, prev_prev_e);
          if (prev_e == 1e9 || e == prev_prev_e) {
            assert(prev_e == 1e9 || e == prev_prev_e);
            sol_i[src].erase(std::next(t).base());
            dst = src;
            prev_e = e, src = src0;
            f = true;
            LOOOG(SOLVER, TRACE, "Send Route: %s -> %s@epoch%d", dev_names[src].c_str(), dev_names[dst].c_str(), prev_e);
            break;
          }
        }
        assert(f);
      } while (!node_info[src].copy); // Only find the last node to copy to form a tree
      LOOOG(SOLVER, TRACE, "Algo send: %s -> %s@epoch%d", dev_names[src].c_str(), dev_names[dev].c_str(), prev_e);
      assert(topo->devices[src]->type == NODE_GPU && topo->devices[dev]->type == NODE_GPU);
      int src_gpu = topo->devices[src]->get_local_id();
      int dst_gpu = topo->devices[dev]->get_local_id();
      event_sol->addMILPEvent(src_c, src_gpu, dst_gpu, prev_e);
    }
  }
  event_sol->sortEvents();
  for (auto [src_c, m] : demand) {
    event_sol->setChunkSize(src_c, chunk_size_B);
    for (auto [dev, amount] : m) {
      if (amount == -1) {
        event_sol->addInitDev(src_c, dev);
      }
    }
  }
  return event_sol;
}

constexpr double RATE_EPS = 0.01;
bool DfsRateAlloc(SrcChunk src_c, int node, std::set<int> src_nodes, int last_gpu_node, std::map<int, std::vector<std::tuple<int, int, double>>> &sol_i, double demand_left, int epoch_now, const std::map<int, std::string> &dev_names, std::shared_ptr<TopoInfo> topo_info, std::shared_ptr<EventSolution> event_sol, std::map<int, bool> &vis) {
  if (src_nodes.count(node)) {
    return true;
  }
  // (epoch, src, amount)
  std::vector<std::tuple<int, int, double>> &sources = sol_i[node];
  std::sort(sources.begin(), sources.end(), [](const std::tuple<int, int, double> &a, const std::tuple<int, int, double> &b) {
    return std::get<0>(a) == std::get<0>(b) ? std::get<2>(a) > std::get<2>(b) : std::get<0>(a) > std::get<0>(b);
  });
  auto topo = topo_info->topo;
  for (auto &send : sources) {
    auto [epoch, src, amount] = send;
    if (amount < RATE_EPS || vis[src]) {
      continue;
    }
    // Do not allow multiple links between two nodes here
    assert(topo_info->link_info[src][node].size() == 1);
    auto link = topo_info->link_info[src][node][0];
    int prev_e = link.get_prev_epoch(epoch_now);
    if (epoch != prev_e) {
      continue;
    }
    double rate = std::min(demand_left, amount);
    demand_left -= rate;
    send = {epoch, src, amount - rate};
    bool fnd = false;
    if (topo->devices[src]->type == NODE_GPU) {
      assert(topo->devices[src]->type == NODE_GPU && topo->devices[last_gpu_node]->type == NODE_GPU);
      vis[src] = true;
      fnd = DfsRateAlloc(src_c, src, src_nodes, src, sol_i, rate, epoch, dev_names, topo_info, event_sol, vis);
      vis[src] = false;
      if (fnd) {
        assert(dev_names.count(src) && dev_names.count(last_gpu_node) && dev_names.count(node));
        LOOOG(SOLVER, TRACE, "Send route: %s -> %s: %f", dev_names.at(src).c_str(), dev_names.at(node).c_str(), rate);
        LOOOG(SOLVER, TRACE, "Algo send: %s -> %s: %f", dev_names.at(src).c_str(), dev_names.at(last_gpu_node).c_str(), rate);
        int src_gpu = topo->devices[src]->get_local_id();
        int dst_gpu = topo->devices[last_gpu_node]->get_local_id();
        event_sol->addLPEvent(src_c, src_gpu, dst_gpu, epoch, rate);
      } else {
        send = {epoch, src, amount};
        demand_left += rate;
      }
    } else {
      vis[src] = true;
      fnd = DfsRateAlloc(src_c, src, src_nodes, last_gpu_node, sol_i, rate, epoch, dev_names, topo_info, event_sol, vis);
      vis[src] = false;
      if (fnd) {
        assert(dev_names.count(src) && dev_names.count(node));
        LOOOG(SOLVER, TRACE, "Send route: %s -> %s: %f", dev_names.at(src).c_str(), dev_names.at(node).c_str(), rate);
      } else {
        send = {epoch, src, amount};
        demand_left += rate;
      }
    }
    if (demand_left < RATE_EPS) {
      break;
    }
  }
  return demand_left < RATE_EPS;
}

std::set<int> getSrcDev(const SrcChunk &src_c, const LPDemand &demand) {
  assert(demand.count(src_c));
  std::set<int> src_devs;
  for (auto [dev, amount] : demand.at(src_c)) {
    if (demand.at(src_c).at(dev) < 0) {
      src_devs.insert(dev);
    }
  }
  assert(!src_devs.empty());
  return src_devs;
}

// Translate a solution into concrete sends
// The solution may split the chunks for LP due to TE-CCL's modeling, since we perform chunk splitting and part of routing in the sketch phase, we doesn't allow splitting of chunks here and instead use the most dominant path for every chunk.
std::shared_ptr<EventSolution> AlgoSolverBase::extractLPEventSolution(std::shared_ptr<LPSolution> solution, std::shared_ptr<TopoInfo> topo_info, LPDemand &demand, ChunkSize chunk_size_B) {
  auto event_sol = std::make_shared<EventSolution>();
  auto sol = solution->sol;
  auto dev_names = topo_info->dev_names;
  auto node_info = topo_info->node_info;
  std::set<SrcChunk> real_src_chunks;
  // Solve for every source chunk
  for (auto [src_c, epoches] : sol) {
    auto src_devs = getSrcDev(src_c, demand);
    // (dev -> (epoch, src, amount))
    std::map<int, std::vector<std::tuple<int, int, double>>> sol_i;
    // (dev -> (epoch, consumed)
    std::map<int, std::map<int, double>> consumed;
    for (auto [e, links] : epoches) {
      for (auto [link, amount] : links) {
        auto [src, dst, id] = link;
        assert(dev_names.count(src) && dev_names.count(dst));
        LOOOG(SOLVER, TRACE, "Send in sol: %s -> %s: %f", dev_names[src].c_str(), dev_names[dst].c_str(), amount);
        sol_i[dst].push_back({e, src, amount});
      }
    }

    for (auto [epoch, consumes] : solution->consume[src_c]) {
      for (auto [node, amount] : consumes) {
        consumed[node][epoch] = amount;
      }
    }

    for (auto [dev, amount] : demand[src_c]) {
      if (amount <= 0) {
        continue;
      }
      auto [gpu_id, _] = src_c;
      auto src_devs = getSrcDev(src_c, demand);
      int dst_gpu = topo_info->topo->devices[dev]->get_local_id();
      SrcChunk real_src_c = {gpu_id, dst_gpu};
      real_src_chunks.insert(real_src_c);
      double demand_left = demand[src_c][dev];
      for (auto [e, c] : consumed[dev]) {
        demand_left -= c;
        std::map<int, bool> vis;
        vis[dev] = true;
        bool fnd = DfsRateAlloc(real_src_c, dev, src_devs, dev, sol_i, c, e, dev_names, topo_info, event_sol, vis);
        assert(fnd);
      }
      assert(demand_left < RATE_EPS);
    }

  }
  event_sol->sortEvents();
  // merge sends in adjacent epochs
  for (auto &[src_c, events] : event_sol->events) {
    // link -> epoch -> sends
    std::map<std::pair<int, int>, std::vector<std::pair<int, double>>> link_sends;
    for (auto event : events) {
      int src = event.src_gpu, dst = event.dst_gpu;
      link_sends[{src, dst}].emplace_back(event.epoch, event.amount);
    }
    events.clear();
    for (auto &[link, epoch_sends] : link_sends) {
      for (int i = 1; i < epoch_sends.size(); ++i) {
        auto [epoch, amount] = epoch_sends[i];
        auto [prev_epoch, prev_amount] = epoch_sends[i - 1];
        if (prev_amount > RATE_EPS) {
          epoch_sends[i] = {prev_epoch, amount + prev_amount};
          epoch_sends[i - 1] = {0, 0};
        }
      }
    }
    for (auto &[link, epoch_sends] : link_sends) {
      for (auto [epoch, amount] : epoch_sends) {
        if (amount > RATE_EPS) {
          LOOOG(SOLVER, TRACE, "Adding Send: %d -> %d: %.2lf at #%lld", link.first, link.second, amount, epoch);
          event_sol->addLPEvent(src_c, link.first, link.second, epoch, amount);
        }
      }
    }
  }
  event_sol->sortEvents();
  for (auto &[src_c, events] : event_sol->events) {
    for (auto &event : events) {
      // event.epoch = events[0].epoch;
    }
  }
  // WARNING: We don't do raw solves so we don't need to set chunk size and init devs
  /*
  for (auto src_c : real_src_chunks) {
    event_sol->setChunkSize(src_c, chunk_size_B);
    auto [src_gpu, idx] = src_c;
    auto devs = getSrcDev({src_gpu, -1}, demand);
    for (auto dev : devs) {
      event_sol->addInitDev(src_c, dev);
    }
  }
  */
  return event_sol;
}

SolverConfig::SolverConfig(nlohmann::json config) {
  split_chunks = config.value("split_chunks", 1);
  chunk_size_B = config.value("chunk_size_B", 0);
  alpha_threshold = config.value("alpha_threshold", 0.1);
  trial_time_factor = config.value("trial_time_factor", 1.0);
  epoch_time_strat = EPOCH_TIME_STRAT_MAP.at(config.value("epoch_time_strat", "FASTEST_LINK"));
  epoch_model = EPOCH_MODEL_MAP.at(config.value("epoch_model", "ACCURATE"));
  model_extra_sends_big_beta = config.value("model_extra_sends_big_beta", false);
  model_extra_sends_small_beta = config.value("model_extra_sends_small_beta", false);
  balance_ratio = config.value("balance_ratio", 0.1);
  alpha_epoch_duration_ratio_max = config.value("alpha_epoch_duration_ratio_max", 100);
}