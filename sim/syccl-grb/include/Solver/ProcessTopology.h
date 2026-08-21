// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sketch/Topology.h"
#include "PerfModel/Topology.h"
#include "PerfModel/Link.h"
#include "Solver/common.h"
#include <cassert>
#include <memory>
#include <tuple>
#include <vector>

namespace Solver {

struct LinkInfo {
  // In terms of epoches
  // occupy time is how long a chunk send will occupy the link, counts towards capacity calculation
  // propagation latency is how long does it take for the next hop to start forwarding the chunk, possibly pipeling in the network.
  // However, the chunk may not completely arrive after this latency. full latency is the time it takes for the chunk to completely arrive at the destination.
  double bw_Bpns, lat_ns, epoch_duration_ns;

  double alpha, beta;
  ChunkSize chunk_size_B;

  EpochModel epoch_model;
  LinkType type;

  int occupy_e, propagation_lat_e, full_lat_e;
  double occupy_t, propagation_t, full_t;
  double capacity_chunk_double;
  int capacity_chunk_int;
  LinkInfo(double bw_Bpns, double lat_ns, double epoch_duration_ns, ChunkSize chunk_size_B, EpochModel epoch_model, LinkType type);
  LinkInfo(double bw_Bpns, double lat_ns);
  LinkInfo() = default;
  void updateSpec(double _epoch_duration_ns, ChunkSize _chunk_size_B, EpochModel _epoch_model, LinkType _type);
  std::string toStr() const {
    auto s = format("bw: %.2lf, lat: %.2lf, epoch_duration: %.2lf, chunk_size: %d, alpha: %.2lf, beta: %.2lf, occupy_e: %d, prop_e: %d, full_e: %d, cap: %.2lf", bw_Bpns, lat_ns, epoch_duration_ns, chunk_size_B, alpha, beta, occupy_e, propagation_lat_e, full_lat_e, capacity_chunk_double);
    return s;
  }
};

// The solver need to know which nodes copy chunk to generate the solution
struct NodeInfo {
  bool buffer, copy, recv, sw_hyperedge;
  int sw_level;
  node_type_e type;
  NodeInfo(node_type_e type, int level, bool sw_to_hyperedge, bool nic_to_hyperedge);
  NodeInfo() = default;
};

struct HyperEdgeInfo {
  int src, dst;
  // Aggregated lat and minimum bandwidth
  double bw, lat;
  // used links in the original topology for this hyperedge
  std::vector<LinkInfoId> link_groups;
  HyperEdgeInfo() = default;
  HyperEdgeInfo(const HyperEdgeInfo &other) {
    bw = other.bw, lat = other.lat;
    src = other.src, dst = other.dst;
    // deep copy
    link_groups = other.link_groups;
  }
  std::string toStr() const {
    auto s = format("HyperEdge: %d -> %d, bw: %.2lf, lat: %.2lf, hops:", src, dst, bw, lat);
    for (auto [src, dst, idx] : link_groups) {
      s += format(" (%d->%d#%d)", src, dst, idx);
    }
    return s;
  }
};

struct ModeledLink {
  LinkInfo info;
  HyperEdgeInfo hyperedge_info;
  bool is_hyperedge;
  bool replace_by_hyperedge;
  // model x->switch forwarding as instant while switch->x as normal link, account double delay on sw->x
  LinkType type;
  LinkInfoId id;
  ModeledLink() = default;
  ModeledLink(LinkInfo info, bool replace_by_hyperedge, LinkType type) : info(info), is_hyperedge(false), replace_by_hyperedge(replace_by_hyperedge), type(type) {}
  ModeledLink(LinkInfo info, HyperEdgeInfo hyperedge_info, LinkType type) : info(info), hyperedge_info(hyperedge_info), is_hyperedge(true), replace_by_hyperedge(false), type(type) {}
  int get_prev_epoch(int e);
  int get_prev_epoch_full(int e);
  bool present(bool use_hyperedge);
  std::string toStr() const {
    auto [src, dst, idx] = id;
    std::string s = format("Link %d->%d #%d: \n", src, dst, idx);
    s += "Info: " + info.toStr() + "\n";
    if (is_hyperedge) {
      s += "HyperEdge: " + hyperedge_info.toStr();
    } else {
      if (replace_by_hyperedge) {
        s += "Replaced by HyperEdge";
      } else {
        s += "Normal link";
      }
    }
    return s;
  }
  std::string toStr(const std::map<int, std::string> &dev_names) const {
    auto [src, dst, idx] = id;
    std::string s = format("Link %s->%s #%d: \n", dev_names.at(src).c_str(), dev_names.at(dst).c_str(), idx);
    s += "Info: " + info.toStr() + "\n";
    if (is_hyperedge) {
      s += "HyperEdge: " + hyperedge_info.toStr();
    } else {
      if (replace_by_hyperedge) {
        s += "Replaced by HyperEdge";
      } else {
        s += "Normal link";
      }
    }
    return s;
  }
};

// used for solver modeling
class TopoInfo {
public:
  std::shared_ptr<Simulation::Topology> topo;
  TopoConfig config;
  int chunk_size_B;
  std::map<int, NodeInfo> node_info;
  std::map<int, std::string> dev_names;
  std::map<int, std::map<int, std::vector<ModeledLink>>> link_info;
  std::map<std::shared_ptr<Simulation::Link>, LinkInfoId> linktoid;
  TopoInfo(std::shared_ptr<Simulation::Topology> topo, const TopoConfig &topo_config);
  void dfsHyperEdge(int node, HyperEdgeInfo now_e, std::vector<HyperEdgeInfo> &hyperedges, std::map<int, bool> &vis);
  double getEpochTime(ChunkSize chunk_size_B, EpochTimeStrategy strategy, double balance_ratio, double alpha_epoch_duration_ratio_max, EpochModel epoch_model);
  void updateAllLinks(double epoch_duration_ns, ChunkSize chunk_size_B, EpochModel epoch_model);
  int getMaxHopDist(bool count_sw);
  double getSlowestChunkTime(ChunkSize chunk_size_B, int maxHopDist);
};

} // namespace Solver