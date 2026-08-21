// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gurobi_c++.h"
#include "common/json.hpp"
#include "common/log.h"
#include "Sketch/Topology.h"
#include <set>
#include <map>
#include <cmath>
#include <memory>
#include <cstdint>
#include <iostream>

constexpr uint64_t SLICE_SIZE_B = (1 << 19);

#define MULTI_THREAD_ENABLE

#define MAX_THREAD_NUM 128
extern int PARALLEL_THREAD_NUM;
extern int PERFMODEL_POOL_SIZE;
extern int SKETCH_EXPLORATION_THREAD_NUM;

#define USE_SHIFT_ISO true
#define SKIP_ISO_MERGE false

namespace Solver {
// (src, dst, idx)
using LinkInfoId = std::tuple<int, int, int>;
// (src_gpu, pos)
using SrcChunk = std::pair<int, int>;
// demand using deviceID, (Src_chunk, device id) -> demand(-1/0/1)
using MILPDemand = std::map<SrcChunk, std::map<int, int>>;
using ChunkSize = uint64_t;
// src_c, node, e
using BufferVars = std::map<std::tuple<SrcChunk, int, int>, GRBVar>;
// src_c, src, dst, e
using ForwardVars = std::map<std::tuple<SrcChunk, LinkInfoId, int>, GRBVar>;
using ForwardVarsUB = std::map<std::tuple<SrcChunk, LinkInfoId, int>, int>;
// src_c, node, e
using ReceivedVars = std::map<std::tuple<SrcChunk, int, int>, GRBVar>;
// src_c, node, e
using ConsumeVars = std::map<std::tuple<SrcChunk, int, int>, GRBVar>;

const double MODEL_THRES = 0.08;
inline double ceilThres(double x, double thres = MODEL_THRES) {
  if (x - floor(x) < thres) {
    return floor(x);
  } else {
    return ceil(x);
  }
}

inline double floorThres(double x, double thres = MODEL_THRES) {
  if (ceil(x) - x < thres) {
    return ceil(x);
  } else {
    return floor(x);
  }
}

enum MILPSolveAlgo {
  SOLVE_NORMAL,
  SOLVE_BY_SAT
};

enum EpochModel {
  EPOCH_MODEL_TECCL,
  EPOCH_MODEL_ACCURATE,
  EPOCH_MODEL_PIPELINE,
};

const std::map<std::string, EpochModel> EPOCH_MODEL_MAP = {
    {"TECCL", EPOCH_MODEL_TECCL},
    {"ACCURATE", EPOCH_MODEL_ACCURATE},
    {"PIPELINE", EPOCH_MODEL_PIPELINE}
};

const std::map<EpochModel, std::string> EPOCH_MODEL_STR_MAP = {
    {EPOCH_MODEL_TECCL, "TECCL"},
    {EPOCH_MODEL_ACCURATE, "ACCURATE"},
    {EPOCH_MODEL_PIPELINE, "PIPELINE"}
};

enum EpochTimeStrategy {
  EPOCHTIME_STRAT_FASTEST_LINK,
  EPOCHTIME_STRAT_SLOWEST_LINK,
  EPOCHTIME_STRAT_BALANCED,
  EPOCHTIME_STRAT_FULL_DELAY,
  EPOCHTIME_STRAT_BETA_MULT
};

const std::map<std::string, EpochTimeStrategy> EPOCH_TIME_STRAT_MAP = {
    {"FASTEST_LINK", EPOCHTIME_STRAT_FASTEST_LINK},
    {"SLOWEST_LINK", EPOCHTIME_STRAT_SLOWEST_LINK},
    {"BALANCED", EPOCHTIME_STRAT_BALANCED},
    {"FULL_DELAY", EPOCHTIME_STRAT_FULL_DELAY},
    {"BETA_MULT", EPOCHTIME_STRAT_BETA_MULT}
};

enum LinkType {
  LINK_SWITCH_TO_GPU,
  LINK_GPU_TO_SWITCH,
  LINK_SWITCH_TO_SWITCH,
  LINK_GPU_TO_GPU
};

inline bool linkCountBeta(LinkType type) {
  return type == LINK_SWITCH_TO_GPU || type == LINK_GPU_TO_GPU;
}

inline bool isSwNode(node_type_e type) {
  return type == NODE_SWITCH_NV || type == NODE_SWITCH_NET || type == NODE_NIC;
}

inline bool srcisGPU(LinkType type) {
  return type == LINK_GPU_TO_SWITCH || type == LINK_GPU_TO_GPU;
}

inline bool dstisGPU(LinkType type) {
  return type == LINK_SWITCH_TO_GPU || type == LINK_GPU_TO_GPU;
}

inline LinkType getLinkType(node_type_e src, node_type_e dst) {
  bool src_sw = isSwNode(src), dst_sw = isSwNode(dst);
  if (src_sw && dst_sw) {
    return LINK_SWITCH_TO_SWITCH;
  } else if (src_sw && !dst_sw) {
    return LINK_SWITCH_TO_GPU;
  } else if (!src_sw && dst_sw) {
    return LINK_GPU_TO_SWITCH;
  } else {
    return LINK_GPU_TO_GPU;
  }
}

const std::map<LinkType, std::string> LINK_TYPE_STR_MAP = {
  {LINK_SWITCH_TO_GPU, "SW_TO_GPU"},
  {LINK_GPU_TO_SWITCH, "GPU_TO_SW"},
  {LINK_SWITCH_TO_SWITCH, "SW_TO_SW"},
  {LINK_GPU_TO_GPU, "GPU_TO_GPU"}
};

class SolutionBase {
public:
  int nepoches;
  int nsolves;
  double epoch_duration_ns, finish_time_ns;
  double epoch_probe_time, construct_time, solve_time;
  std::string metatoStr() {
    auto s = format("nepoches=%d, epoch_duration_ns=%f, finish_time_ns=%f\n", nepoches, epoch_duration_ns, finish_time_ns);
    s += format("epoch_probe_time=%fs, construct_time=%fs, solve_time=%fs\n", epoch_probe_time, construct_time, solve_time);
    s += format("nsolves=%d", nsolves);
    return s;
  }
};

using SourceCfg = std::vector<SrcChunk>;
class MILPSolution : public SolutionBase {
public:
  // (Chunk, epoch) -> sends
  std::map<SrcChunk, std::map<int, std::vector<LinkInfoId>>> sol;
};

struct MILPVarsMap {
  BufferVars B;
  ForwardVars F;
  ForwardVarsUB FUB;
  ReceivedVars R;
};

class MILPConfig {
public:
  SourceCfg sources;
  MILPDemand demand;
  MILPSolveAlgo algo = SOLVE_NORMAL;
};

using LPDemand = std::map<SrcChunk, std::map<int, double>>;
class LPSolution : public SolutionBase {
public:
  // (Chunk, epoch) -> sends
  std::map<SrcChunk, std::map<int, std::vector<std::pair<LinkInfoId, double>>>> sol;
  // (Chunk, epoch) -> consume
  std::map<SrcChunk, std::map<int, std::vector<std::pair<int, double>>>> consume;
};

struct LPVarsMap {
  BufferVars B;
  ForwardVars F;
  ConsumeVars C;
  ReceivedVars R;
};

class LPConfig {
public:
  SourceCfg sources;
  LPDemand demand;
};

// specify a subset of the topology
class TopoConfig {
public:
  std::set<int> devices_used;
  bool sw_hyperedge, nic_hyperedge;
  // option to limit hyperedges generated between devices
  bool sw_hyperedge_use_subgraph = false;
  std::set<std::pair<int, int>> subgraph_edges;
};

struct ScheduleEvent {
  int src_gpu, dst_gpu;
  uint64_t epoch;
  double amount;
  bool copy, reduce;
  int layer_used = -1;
  ScheduleEvent(int src_gpu, int dst_gpu, uint64_t epoch, double amount, bool copy, bool reduce) : src_gpu(src_gpu), dst_gpu(dst_gpu), epoch(epoch), amount(amount), copy(copy), reduce(reduce) {}
  void set_layer(int layer) {
    layer_used = layer;
  }
  std::string toStr() {
    return format("%d->%d: %.2lf at #%lld layer %d", src_gpu, dst_gpu, amount, epoch, layer_used);
  }
  nlohmann::json to_json();
};

using EventTime = uint64_t;
class EventSolution {
public:
  std::map<SrcChunk, std::vector<ScheduleEvent>> events;
  std::map<SrcChunk, std::set<int>> init_devs;
  std::map<SrcChunk, ChunkSize> chunk_sizes;
  // SrcChunk here shoud be a representative of the reduce tree
  void addReduceEvent(SrcChunk src_c, int src, int dst, uint64_t e) {
    events[src_c].push_back(ScheduleEvent(src, dst, e, 1, false, true));
  }
  void addMILPEvent(SrcChunk src_c, int src, int dst, uint64_t e) {
    events[src_c].push_back(ScheduleEvent(src, dst, e, 1, true, false));
  }
  void addLPEvent(SrcChunk src_c, int src, int dst, uint64_t e, double amount) {
    events[src_c].push_back(ScheduleEvent(src, dst, e, amount, false, false));
  }
  // device id, not gpu id
  void addInitDev(SrcChunk src_c, int dev) {
    init_devs[src_c].insert(dev);
  }
  void setChunkSize(SrcChunk src_c, ChunkSize size) {
    chunk_sizes[src_c] = size;
  }
  void sortEvents() {
    for (auto &[src_c, evs] : events) {
      std::sort(evs.begin(), evs.end(), [](const ScheduleEvent &a, const ScheduleEvent &b) {
        return a.epoch < b.epoch;
      });
    }
  }
  std::string toStr() {
    std::string s = "EventSolution: \n";
    for (auto [src_c, evs] : events) {
      s += format("\nSrcChunk(%d, %d): ", std::get<0>(src_c), std::get<1>(src_c));
      for (auto &ev : evs) {
        s += ev.toStr() + ", ";
      }
    }
    s += "\nInit Devs: \n";
    for (auto &[src_c, devs] : init_devs) {
      s += format("SrcChunk(%d, %d) init devs: ", std::get<0>(src_c), std::get<1>(src_c));
      for (auto dev : devs) {
        s += std::to_string(dev) + ", ";
      }
      s += "; ";
    }
    s += "\nChunk Sizes: \n";
    for (auto &[src_c, size] : chunk_sizes) {
      s += format("SrcChunk(%d, %d) size: %llu; ", std::get<0>(src_c), std::get<1>(src_c), size);
    }
    return s;
  }
  nlohmann::json to_json();
  static std::shared_ptr<EventSolution> from_json(nlohmann::json &j);
};

class SimResult {
public:
    std::shared_ptr<EventSolution> schedule = nullptr;
    // gpuid -> received chunks
    std::map<int, std::vector<std::set<SrcChunk>>> received_chunks;
    // (SrcChunk, src -> dst) -> received time
    std::map<std::tuple<SrcChunk, int, int>, std::vector<EventTime>> slice_received_time, slice_sent_time;
    double time_us = 0;
    void addSliceSentTime(SrcChunk src_chunk, int src, int dst, EventTime send_time) {
        slice_sent_time[{src_chunk, src, dst}].push_back(send_time);
    }
    void addSliceReceivedTime(SrcChunk src_chunk, int src, int dst, EventTime recv_time) {
        slice_received_time[{src_chunk, src, dst}].push_back(recv_time);
    }
    std::string getReceived() {
      std::string res;
      for (auto &[gpuid, chunks] : received_chunks) {
          res += "GPU" + std::to_string(gpuid) + ": ";
          for (auto &rc : chunks) {
              std::string chunk_str;
              for (auto &[gpuid, pos] : rc) {
                  assert(gpuid >= 0);
                  assert(pos >= 0);
                  chunk_str += format("%d#%d+", gpuid, pos);
              }
              res += chunk_str + ", ";
          }
          res += "\n";
      }
      return res;
    }
    std::string toStr(bool verbose = true) {
        std::string res = "SimResult:\n";
        res += "Schedule: " + schedule->toStr() + "\n";
        for (auto &[gpuid, chunks] : received_chunks) {
            res += "GPU" + std::to_string(gpuid) + " received: ";
            for (auto &rc : chunks) {
                std::string chunk_str;
                for (auto &[gpuid, pos] : rc) {
                    assert(gpuid >= 0);
                    assert(pos >= 0);
                    chunk_str += format("%d#%d+", gpuid, pos);
                }
                res += chunk_str + ", ";
            }
            res += "\n";
        }
        if (verbose) {
          for (auto &[send, recv_time] : slice_received_time) {
              auto &send_time = slice_sent_time[send];
              assert(send_time.size() == recv_time.size());
              auto &[src_chunk, src, dst] = send;
              auto &[gpuid, pos] = src_chunk;
              res += "Send " + format("%d#%d from %d -> %d: ", gpuid, pos, src, dst);
              for (int i = 0; i < send_time.size(); ++i) {
                  res += format("s:%lld -> r:%lld, ", send_time[i], recv_time[i]);
              }
              res += "\n";
          }
        }
        res += "Compeletion Time: " + std::to_string(time_us) + "us";
        return res;
    }
    nlohmann::json to_json();
};

} // namespace Solver