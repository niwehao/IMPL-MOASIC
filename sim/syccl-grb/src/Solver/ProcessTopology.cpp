// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "common/CLI11.hpp"
#include "PerfModel/Topology.h"
#include "PerfModel/Device.h"
#include "PerfModel/Link.h"
#include "Solver/Vars.hpp"
#include "Solver/common.h"
#include "Solver/ProcessTopology.h"
#include "Sketch/common.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

using namespace Solver;

NodeInfo::NodeInfo(node_type_e type, int level, bool sw_to_hyperedge, bool nic_to_hyperedge): type(type) {
  if (type == NODE_SWITCH_NV || type == NODE_SWITCH_NET || type == NODE_NIC) {
    assert(level != -1);
  }
  // TODO: assuming all nodes have buffer may speedup the process while not affecting solution quality?
  buffer = (type == NODE_GPU);
  copy = (type == NODE_GPU);
  // TODO: according to demand?
  recv = (type == NODE_GPU);
  sw_hyperedge = (sw_to_hyperedge && (type == NODE_SWITCH_NV || type == NODE_SWITCH_NET)) ||
                 (nic_to_hyperedge && type == NODE_NIC);
  sw_level = level;
}

LinkInfo::LinkInfo(double bw_Bpns, double lat_ns, double epoch_duration_ns, ChunkSize chunk_size_B, EpochModel epoch_model, LinkType type): bw_Bpns(bw_Bpns), lat_ns(lat_ns), epoch_duration_ns(epoch_duration_ns), chunk_size_B(chunk_size_B), epoch_model(epoch_model), type(type) {
  updateSpec(epoch_duration_ns, chunk_size_B, epoch_model, type);
}

LinkInfo::LinkInfo(double bw_Bpns, double lat_ns): bw_Bpns(bw_Bpns), lat_ns(lat_ns) {
}

int ModeledLink::get_prev_epoch(int e) {
  return e - info.propagation_lat_e;
}

int ModeledLink::get_prev_epoch_full(int e) {
  return e - info.full_lat_e;
}

bool ModeledLink::present(bool use_hyperedge) {
  if (use_hyperedge && replace_by_hyperedge) {
    return false;
  }
  if (!use_hyperedge && is_hyperedge) {
    return false;
  }
  return true;
}

constexpr double ALPHA_THRES = 0.1, FULL_THRES = 0;
constexpr uint64_t PIPELINE_CHUNK_SIZE_B = SLICE_SIZE_B;
void LinkInfo::updateSpec(double _epoch_duration_ns, ChunkSize _chunk_size_B, EpochModel _epoch_model, LinkType _type) {
  epoch_duration_ns = _epoch_duration_ns;
  chunk_size_B = _chunk_size_B;
  epoch_model = _epoch_model;
  type = _type;
  alpha = lat_ns, beta = chunk_size_B / bw_Bpns;
  if (epoch_model == EPOCH_MODEL_TECCL) {
    // alpha-beta model in the paper where a_e and b_e are separated
    int a_e = ceil(alpha / epoch_duration_ns), b_e = ceil(beta / epoch_duration_ns);
    if (alpha / epoch_duration_ns < ALPHA_THRES) {
      a_e = 0;
    }
    b_e = std::max(0, b_e - 1) + 1;
    occupy_t = beta;
    occupy_e = b_e;
    if (!linkCountBeta(type)) {
      full_t = alpha, full_lat_e = a_e;
    } else {
      full_t = alpha + beta, full_lat_e = a_e + b_e;
    }
    propagation_t = full_t, propagation_lat_e = full_lat_e;
  } else if (epoch_model == EPOCH_MODEL_ACCURATE) {
    occupy_t = beta;
    occupy_e = ceil(beta / epoch_duration_ns);
    if (!linkCountBeta(type)) {
      full_t = alpha, full_lat_e = ceilThres(full_t / epoch_duration_ns, ALPHA_THRES);
    } else {
      full_t = alpha + beta, full_lat_e = ceilThres(full_t / epoch_duration_ns, FULL_THRES);
    }
    propagation_t = full_t, propagation_lat_e = full_lat_e;
  } else if (epoch_model == EPOCH_MODEL_PIPELINE) {
    occupy_t = beta;
    occupy_e = ceil(beta / epoch_duration_ns);
    double first_slice_beta = std::min(chunk_size_B, PIPELINE_CHUNK_SIZE_B) / bw_Bpns;
    if (!linkCountBeta(type)) {
      full_t = alpha, full_lat_e = ceilThres(full_t / epoch_duration_ns, ALPHA_THRES);
      propagation_t = alpha, propagation_lat_e = ceilThres(propagation_t / epoch_duration_ns, ALPHA_THRES);
    } else {
      full_t = alpha + beta, full_lat_e = ceilThres(full_t / epoch_duration_ns, FULL_THRES);
      propagation_t = alpha + first_slice_beta, propagation_lat_e = ceilThres(propagation_t / epoch_duration_ns, FULL_THRES);
    }
  } else {
    assert(false);
  }
  // in case this is zero, occupy_t should be greater than one and the constraint would be modified accordingly to make sense
  capacity_chunk_double = epoch_duration_ns / beta;
  capacity_chunk_int = floor(capacity_chunk_double);
  assert((capacity_chunk_int == 0 && occupy_e > 1) || (occupy_e == 1 && capacity_chunk_int > 0));
  capacity_chunk_int = std::max(1, capacity_chunk_int);
}

void TopoInfo::dfsHyperEdge(int node, HyperEdgeInfo now_e, std::vector<HyperEdgeInfo> &hyperedges, std::map<int, bool> &vis) {
  assert(node_info[node].sw_hyperedge);
  auto dev = topo->devices[node];
  for (auto [nxt, link] : dev->links) {
    if (nxt == node || config.devices_used.count(nxt) == 0) {
      continue;
    }
    HyperEdgeInfo nxt_e(now_e);
    nxt_e.bw = std::min(nxt_e.bw, link->getBandwidthBpns());
    nxt_e.lat += link->getLatencyns();
    nxt_e.link_groups.push_back(linktoid[link]);
    nxt_e.dst = nxt;
    if (!node_info[nxt].sw_hyperedge) {
      if (!vis[nxt]) {
        hyperedges.push_back(nxt_e);
      }
      // no further Dfs from normal nodes
    } else {
      if (!vis[nxt]) {
        vis[nxt] = true;
        dfsHyperEdge(nxt, nxt_e, hyperedges, vis);
        vis[nxt] = false;
      }
    }
  }
}

double TopoInfo::getEpochTime(ChunkSize chunk_size_B, EpochTimeStrategy strategy, double balance_ratio, double alpha_epoch_duration_ratio_max, EpochModel epoch_model) {
  bool use_hyperedge = config.sw_hyperedge || config.nic_hyperedge;
  std::vector<double> a, b, prop, full;
  for (auto [src, smap] : link_info) {
    for (auto [dst, links] : smap) {
      for (auto &link : links) {
        if (!link.present(use_hyperedge)) {
          continue;
        }
        link.info.updateSpec(1, chunk_size_B, epoch_model, link.type);
        if (link.info.alpha != 0) {
          a.push_back(link.info.alpha);
        }
        b.push_back(link.info.beta);
        if (linkCountBeta(link.type)) {
          prop.push_back(link.info.propagation_t);
          full.push_back(link.info.full_t);
        }
      }
    }
  }
  double final_epoch_time = 0;
  double mina = *std::min_element(a.begin(), a.end());
  double minb = *std::min_element(b.begin(), b.end());
  double mprop = *std::max_element(prop.begin(), prop.end());
  double mfull = *std::max_element(full.begin(), full.end());
  if (strategy == EPOCHTIME_STRAT_FASTEST_LINK) {
    final_epoch_time = minb;
  } else if (strategy == EPOCHTIME_STRAT_SLOWEST_LINK) {
    final_epoch_time = *std::max_element(b.begin(), b.end());
  } else if (strategy == EPOCHTIME_STRAT_BALANCED) {
    // double cand = std::min(mina, minb);
    double cand = minb;
    // when alpha is large
    while (cand / mfull < balance_ratio) {
      cand += minb;
    }
    final_epoch_time = cand;
    /*
    if (cand / mina < balance_ratio) {
      // using beta, alpha dominating over beta will cause propagation epoch being too long
      double ret = cand;
      while (ret / mina < balance_ratio) {
        ret += cand;
      }
      final_epoch_time = ret;
    } else if (cand / minb < balance_ratio) {
      final_epoch_time = minb;
    } else {
      final_epoch_time = cand;
    }
    */
  } else if (strategy == EPOCHTIME_STRAT_FULL_DELAY) {
    final_epoch_time = *std::min_element(prop.begin(), prop.end());
    final_epoch_time *= ceil(balance_ratio);
  } else if (strategy == EPOCHTIME_STRAT_BETA_MULT) {
    
    double ret = minb;
    if (ret / mfull < balance_ratio) {
      while (ret / mfull < balance_ratio) {
        ret += minb;
      }
      final_epoch_time = ret;
    } else {
      int x = 1;
      double cand = (minb / (x + 1));
      while (cand / mfull > balance_ratio) {
        x++;
        cand = (minb / (x + 1));
      }
      final_epoch_time = minb / x;
    }
  } else {
    assert(false);
  }
  double min_alpha = *std::min_element(a.begin(), a.end());
  if (min_alpha / final_epoch_time > alpha_epoch_duration_ratio_max) {
    final_epoch_time = min_alpha / alpha_epoch_duration_ratio_max;
  }
  LOOOG(SOLVER, INFO, "Final Epoch Time: %lf, minb: %lf, mina: %lf, mfull: %lf, mprop: %lf", final_epoch_time, *std::min_element(b.begin(), b.end()), *std::min_element(a.begin(), a.end()), *std::min_element(full.begin(), full.end()), *std::min_element(prop.begin(), prop.end()));
  return final_epoch_time;
}

void TopoInfo::updateAllLinks(double epoch_duration_ns, ChunkSize chunk_size_B, EpochModel epoch_model) {
  LOOOG(SOLVER, TRACE, "Update all links with epoch_duration_ns: %lf, chunk_size_B: %d, epoch_model: %s", epoch_duration_ns, chunk_size_B, EPOCH_MODEL_STR_MAP.at(epoch_model).c_str());
  for (auto &[src, smap] : link_info) {
    for (auto &[dst, links] : smap) {
      for (auto &link : links) {
        link.info.updateSpec(epoch_duration_ns, chunk_size_B, epoch_model, link.type);
        // LOOOG(SOLVER, TRACE, "Updated Link: %s", link.toStr().c_str());
      }
    }
  }
}

int TopoInfo::getMaxHopDist(bool count_sw) {
  bool use_hyperedge = config.sw_hyperedge || config.nic_hyperedge;
  std::map<int, int> dev_local_id;
  int nd = 0;
  for (auto i : config.devices_used) {
    dev_local_id[i] = nd++;
  }
  // Floyd-Warshell
  std::vector<std::vector<int>> dist(nd, std::vector<int>(nd, 1e9));
  for (int i = 0; i < nd; ++i) {
    dist[i][i] = 0;
  }
  for (auto &[src, smap] : link_info) {
    for (auto &[dst, links] : smap) {
      for (auto &link : links) {
        if (!link.present(use_hyperedge)) {
          continue;
        }
        if (!count_sw) {
          dist[dev_local_id[src]][dev_local_id[dst]] = linkCountBeta(link.type) ? 1 : 0;
        } else {
          dist[dev_local_id[src]][dev_local_id[dst]] = 1;
        }
      }
    }
  }
  for (int k = 0; k < nd; ++k) {
    for (int i = 0; i < nd; ++i) {
      for (int j = 0; j < nd; ++j) {
        dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
      }
    }
  }
  int max_dist = 0;
  LOOOG(SOLVER, TRACE, "Hop Dist Matrix:");
  for (int i = 0; i < nd; ++i) {
    std::string s = "";
    for (int j = 0; j < nd; ++j) {
      s += std::to_string(dist[i][j]) + " ";
    }
    LOOOG(SOLVER, TRACE, s.c_str());
  }
  for (int i = 0; i < nd; ++i) {
    for (int j = 0; j < nd; ++j) {
      if (dist[i][j] == 1e9) {
        continue;
      }
      max_dist = std::max(max_dist, dist[i][j]);
    }
  }
  LOOOG(SOLVER, INFO, "count_sw: %d, max_hop_dist: %d", count_sw, max_dist);
  return max_dist;
}

double TopoInfo::getSlowestChunkTime(ChunkSize chunk_size_B, int maxHopDist) {
  bool use_hyperedge = config.sw_hyperedge || config.nic_hyperedge;
  double slowest = 0;
  for (auto &[src, smap] : link_info) {
    for (auto &[dst, links] : smap) {
      for (auto &link : links) {
        if (!link.present(use_hyperedge)) {
          continue;
        }
        double t;
        if (link.type == LINK_SWITCH_TO_GPU) {
          t = link.info.lat_ns * maxHopDist + chunk_size_B / link.info.bw_Bpns;
        } else {
          t = link.info.lat_ns + chunk_size_B / link.info.bw_Bpns;
        }
        slowest = std::max(slowest, t);
      }
    }
  }
  return slowest;
}

TopoInfo::TopoInfo(std::shared_ptr<Simulation::Topology> topo, const TopoConfig &topo_config): topo(topo), config(topo_config) {
  std::map<int, int> sw_layers;
  for (auto &layer : topo->topo_info.layer_info) {
    for (auto &group : layer.groups) {
      // for (auto &dev : group.devices) {
        if (group.switch_device != nullptr) {
          int sw_id = group.switch_device->get_id();
          sw_layers[sw_id] = group.layer_id;
        }
      // }
    }
  }
  for (auto [devid, layer] : sw_layers) {
    auto dev = topo->devices[devid];
    LOOOG(SOLVER, TRACE, "Sw %s in layer %d", dev->toStr().c_str(), layer);
  }
  // process nodeinfo and linkinfo
  for (auto i : config.devices_used) {
    if (!sw_layers.count(i)) {
      sw_layers[i] = -1;
    }
    auto dev = topo->devices[i];
    dev_names[i] = dev->toStr();
    node_info[i] = NodeInfo(dev->type, sw_layers[i], config.sw_hyperedge, config.nic_hyperedge);
  }
  for (auto i : config.devices_used) {
    auto dev = topo->devices[i];
    for (auto &[nxt, link] : dev->links) {
      if (nxt == i || !config.devices_used.count(nxt)) {
        continue;
      }
      auto bw_Bpns = link->getBandwidthBpns(), lat_ns = link->getLatencyns();
      auto info = LinkInfo(bw_Bpns, lat_ns);
      bool replace_by_hyperedge = (node_info[i].sw_hyperedge || node_info[nxt].sw_hyperedge);
      LinkType type = getLinkType(node_info[i].type, node_info[nxt].type);
      link_info[i][nxt].emplace_back(info, replace_by_hyperedge, type);
      linktoid[link] = {i, nxt, (int)link_info[i][nxt].size() - 1};
    }
  }
  bool gen_hyperedge = config.sw_hyperedge || config.nic_hyperedge;
  if (gen_hyperedge) {
    // for every non-hyperedge node, translate real edges passing through hyperedge switches into hyperedges
    std::map<int, bool> vis;
    for (auto i : config.devices_used) {
      auto dev = topo->devices[i];
      if (node_info[i].sw_hyperedge) {
        continue;
      }
      for (auto &[nxt, link] : dev->links) {
        if (nxt == i || !config.devices_used.count(nxt)) {
          continue;
        }
        if (node_info[nxt].sw_hyperedge) {
          std::map<int, bool> vis;
          HyperEdgeInfo init_e;
          double bw = link->getBandwidthBpns();
          double latency = link->getLatencyns();
          init_e.bw = bw, init_e.lat = latency, init_e.src = i, init_e.dst = nxt;
          init_e.link_groups.push_back(linktoid[link]);
          vis[i] = true, vis[nxt] = true;
          std::vector<HyperEdgeInfo> gen_hyperedges;
          dfsHyperEdge(nxt, init_e, gen_hyperedges, vis);
          for (auto &he : gen_hyperedges) {
            int src = he.src, dst = he.dst;
            if (config.sw_hyperedge_use_subgraph &&
                config.subgraph_edges.count({src, dst}) == 0) {
              continue;
            }
            // Don't use GPU->NIC->GPU communication
            if (he.link_groups.size() == 2) {
              auto [src, mid1, idx1] = he.link_groups[0];
              auto [mid2, dst, idx2] = he.link_groups[1];
              assert(mid1 == mid2);
              if (topo->devices[src]->type == NODE_GPU && topo->devices[mid1]->type == NODE_NIC && topo->devices[dst]->type == NODE_GPU) {
                continue;
              }
            }
            auto info = LinkInfo(he.bw, he.lat);
            // construct a Modeled hyperedge
            LinkType type = getLinkType(node_info[src].type, node_info[dst].type);
            link_info[src][dst].emplace_back(info, he, type);
          }
        }
      }
    }
  }
  // generate linkinfo id
  // WARNING: vector defaults to deep copying
  for (auto &[src, smap] : link_info) {
    for (auto &[dst, links] : smap) {
      assert(config.devices_used.count(src) && config.devices_used.count(dst));
      for (int i = 0; i < links.size(); ++i) {
        links[i].id = {src, dst, i};
        // LOOOG(SOLVER, TRACE, "Final Link: %s", links[i].toStr(dev_names).c_str());
      }
    }
  }
}