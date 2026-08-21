// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/CommGraph.h"
#include "PerfModel/Event.h"
#include "PerfModel/TopologyLayer.h"
#include "Solver/common.h"
#include "Sketch/Algorithm.h"
#include "PerfModel/PerfModel.h"
#include "PerfModel/Topology.h"
#include "Solver/AlgoSolver.h"
#include "Solver/MIPSolver.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "Sketch/isoSolver.h"
#include "common/ctpl.h"
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <future>

bool demand_map_t::checkValid() {
  std::set<int> gpu_used, chunk_used;
  for (auto &[src, dst] : gpu_map) {
    if (gpu_used.count(dst)) {
      LOOOG(ALGO, INFO, "Invalid GPU map : %s", toStr().c_str());
      return false;
    }
    gpu_used.insert(dst);
  }
  for (auto &[src, dst] : chunk_map) {
    if (chunk_used.count(dst)) {
      LOOOG(ALGO, INFO, "Invalid Chunk map : %s", toStr().c_str());
      return false;
    }
    chunk_used.insert(dst);
  }
  return true;
}

// Add the relevant devices and switches to the topo_config, may not work for all topologies
void addGroup(Solver::TopoConfig &config, const Simulation::topo_info_t &topo_info, int layer_id, int group_id) {
  LOOOG(ALGO, TRACE, "Adding group %d in layer %d", group_id, layer_id);
  assert(layer_id < topo_info.layer_info.size());
  assert(group_id < topo_info.layer_info[layer_id].groups.size());
  auto &layer = topo_info.layer_info[layer_id];
  auto &group = layer.groups[group_id];
  assert(group.layer_id == layer_id && group.group_id == group_id);
  auto &device_used = config.devices_used;
  auto sw = group.switch_device;
  std::set<std::shared_ptr<Simulation::Device>> sws;
  if (sw != nullptr) {
    LOG(ALGO, TRACE, "Switch %s used", sw->toStr().c_str());
    device_used.insert(sw->get_id());
  }
  for (auto &dev : group.devices) {
    LOG(ALGO, TRACE, "Device %s used", dev->toStr().c_str());
    device_used.insert(dev->get_id());
    sws.insert(dev);
  }
  // recurisvely add all the devices required
  std::queue<std::shared_ptr<Simulation::Device>> q;
  for (auto &dev : sws) {
    q.push(dev);
  }
  while (!q.empty()) {
    auto dev = q.front(); q.pop();
    for (int i = layer_id - 1; i >= 0; --i) {
      auto &prev_layer = topo_info.layer_info[i];
      for (auto &prev_group : prev_layer.groups) {
        if (prev_group.switch_device != nullptr &&
            prev_group.switch_device->get_id() == dev->get_id()) {
          for (auto &prev_dev : prev_group.devices) {
            LOG(ALGO, TRACE, "Device %s used", prev_dev->toStr().c_str());
            device_used.insert(prev_dev->get_id());
            if (!sws.count(prev_dev)) {
              q.push(prev_dev);
              sws.insert(prev_dev);
            }
          }
        }
      }
    }
  }
}

void extractLPDemand(intra_group_sched &group_sched, int spilt_chunks, std::map<int, std::vector<int>> &a2a_chunk_parent, std::map<int, std::vector<int>> &a2a_chunk_rem_size, std::shared_ptr<Simulation::Topology> topo, Solver::LPDemand &demand, Solver::SourceCfg &sources) {
  std::set<Solver::SrcChunk> used_cs;
  for (auto &[chunk_id, pairs] : group_sched.src_dest_pairs) {
    for (int i = 0; i < spilt_chunks; ++i) {
      auto &[srcs, dsts] = pairs;
      for (auto dst : dsts) {
        int src = a2a_chunk_parent[chunk_id][dst];
        int src_dev = topo->getNpuDevice(src)->get_id();
        int dst_dev = topo->getNpuDevice(dst)->get_id();
        Solver::SrcChunk chunk = {chunk_id, src};
        if (!used_cs.count(chunk)) {
          sources.push_back(chunk);
        } else {
          used_cs.insert(chunk);
        }
        demand[chunk][dst_dev] = a2a_chunk_rem_size[chunk_id][dst];
        demand[chunk][src_dev] += -a2a_chunk_rem_size[chunk_id][dst];
      }
    }
  }
}

std::shared_ptr<Solver::EventSolution> solveSubgroup(int layer_id, int group_id, Solver::ChunkSize chunk_size_B, int spilt_chunks, double balance_ratio, intra_group_sched &group_sched, std::map<int, std::vector<int>> a2a_chunk_parent, std::map<int, std::vector<int>> a2a_chunk_rem_size, std::shared_ptr<Simulation::Topology> topo, std::shared_ptr<Solver::MIPSolver> solver, bool a2a) {
  if (a2a) assert(spilt_chunks == 1);
  assert(chunk_size_B % spilt_chunks == 0);
  solver->setChunkSize(chunk_size_B / spilt_chunks);
  solver->setBalanceRatio(balance_ratio);
  Solver::TopoConfig topo_config;
  // We only use the relevant sub-topology for solving
  addGroup(topo_config, topo->topo_info, layer_id, group_id);
  topo_config.sw_hyperedge = false;
  // As nic is typically connected to 1-2 GPUs, we translate them to hyperedges
  topo_config.nic_hyperedge = true;
  topo_config.sw_hyperedge_use_subgraph = false;
  std::shared_ptr<Solver::EventSolution> sol;
  if (a2a) {
    Solver::LPConfig lp_config;
    auto &demand = lp_config.demand;
    auto &sources = lp_config.sources;
    LOOOG(ALGO, TRACE, "Solving subgroup Demand: %s", group_sched.toStr().c_str());
    extractLPDemand(group_sched, spilt_chunks, a2a_chunk_parent, a2a_chunk_rem_size, topo, demand, sources);  
    sol = solver->solveLP(lp_config, topo_config);
    LOOOG(ALGO, TRACE, "LP Solution: %s", sol->toStr().c_str());
  } else {
    Solver::MILPConfig milp_config;
    milp_config.algo = Solver::SOLVE_NORMAL;
    auto &demand = milp_config.demand;
    auto &sources = milp_config.sources;
    LOOOG(ALGO, TRACE, "Solving subgroup Demand: %s", group_sched.toStr().c_str());
    for (auto &[chunk_id, pairs] : group_sched.src_dest_pairs) {
      for (int i = 0; i < spilt_chunks; ++i) {
        Solver::SrcChunk chunk = {chunk_id, i};
        auto &[srcs, dsts] = pairs;
        sources.push_back(chunk);
        for (auto src : srcs) {
          int src_dev = topo->getNpuDevice(src)->get_id();
          demand[chunk][src_dev] = -1;
        }
        for (auto dst : dsts) {
          int dst_dev = topo->getNpuDevice(dst)->get_id();
          demand[chunk][dst_dev] = 1;
        }
      }
    }  
    sol = solver->solveMILP(milp_config, topo_config);
    LOOOG(ALGO, TRACE, "MILP Solution: %s", sol->toStr().c_str());
  }
  for (auto &[src_c, evs] : sol->events) {
    for (auto &event : evs) {
      event.set_layer(layer_id);
    }
  }  
  return sol;
}

inline std::mutex pool_mutex, pool_mutex2;
// -1: root, i: isomorphic to ith demand
// Group demands into isomorphic groups with the beginning init_rep_size ones being confirmed non-isomorphic
void merge_isomorphic_demand_groups(std::vector<std::pair<int, intra_group_sched*>>& input_groups, std::shared_ptr<Simulation::Topology> topo, int layer, std::vector<std::pair<int, demand_map_t>> &res, std::set<int> &used_layers, ctpl::thread_pool &pool, bool a2a, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_parent, std::map<intra_group_sched*, std::map<int, std::vector<int>>*> &sched_a2a_chunk_rem_size) {
  int init_rep_size = 1;
  assert(init_rep_size <= input_groups.size());
  res.resize(input_groups.size(), {-1, demand_map_t()});
  if (SKIP_ISO_MERGE) {
    return;
  }
  if (input_groups.size() <= 1) {
    return;
  }

  std::set<int> groups_to_check;
  for (int i = init_rep_size; i < input_groups.size(); i++) {
    groups_to_check.insert(i);
  }

  std::vector<int> isos;
  int cur_isos = 0;
  for (int i = 0; i < init_rep_size; i++) {
    isos.push_back(i);
  }

  while (!groups_to_check.empty()) {
    // check if any remaining demands are isomorphic to exisiting representatives
    while (cur_isos < isos.size()) {
      int id1 = isos[cur_isos++];
      std::vector<int> tmp_non_isomorphic_id;
      std::vector<std::pair<int, demand_map_t>> tmp_isomorphic_id;
/*
      Here we can detect isomorphism by modeling it as an MILP according to the topology and demand
      But this may introduce more overhead than directly solving the demands
      As most of the isomorphic demands are due to our sketch expansion process generating similar demands in each group under a single sketch, we can detect those type of isomorphism using the same mapping as we used in the sketch expansion process
      In other words, this detection method is co-designed with the sketch expansion mapping generation, while it doesn't cover all possible isomorphism, it's enough for our usage and globally it results in a significant speedup
      So other mapping methods/topologies may require different methods
*/
      auto iso_check_func = USE_SHIFT_ISO ? check_isomorphic_demand_simple : nullptr;
      std::vector<std::future<void>> futures;
{  
      std::lock_guard<std::mutex> guard(pool_mutex);
      for (int id2_to_check : groups_to_check) {
#ifdef MULTI_THREAD_ENABLE
        auto future = pool.push([=, &input_groups, &tmp_non_isomorphic_id, &tmp_isomorphic_id, &used_layers, &sched_a2a_chunk_parent, &sched_a2a_chunk_rem_size](int id) {
          iso_check_func(topo, layer, id1, id2_to_check, std::ref(input_groups), std::ref(tmp_non_isomorphic_id), std::ref(tmp_isomorphic_id), std::ref(used_layers), a2a, std::ref(sched_a2a_chunk_parent), std::ref(sched_a2a_chunk_rem_size));
        });
        futures.push_back(std::move(future));
#else
        iso_check_func(topo, layer, id1, id2_to_check, input_groups,tmp_non_isomorphic_id, tmp_isomorphic_id);
#endif
      }
}
#ifdef MULTI_THREAD_ENABLE
{
    // std::lock_guard<std::mutex> guard(pool_mutex);
    // pool.stop(true);
    for (auto &f : futures) {
      f.get();
    }
}
#endif 
      int rep = id1;
      for (auto &[id, dmap] : tmp_isomorphic_id) {
        res[id] = {rep, dmap};
      }
      res[rep] = {-1, demand_map_t()};

      groups_to_check.clear();
      for (auto id : tmp_non_isomorphic_id) {
        groups_to_check.insert(id);
      }
    }
    // After finishing one round of isomorphic check, select one of the remaining groups to be the next representative
    if (!groups_to_check.empty()) {
      isos.push_back(*(groups_to_check.begin()));
      groups_to_check.erase(groups_to_check.begin());
    }
  }
  int reps = 0;
  for (auto &[r, m] : res) {
    reps += r == -1;
  }
  LOOOG(ALGO, INFO, "Merged %d objects to %d isomorphic groups", input_groups.size(), reps);
}

SketchSubDemand::SketchSubDemand(int layer_id, int group_id, intra_group_sched *sched, int split_chunks, double balance_ratio, uint64_t chunk_size_B, std::set<int> used_layers) : layer_id(layer_id), group_id(group_id), sched(sched), split_chunks(split_chunks), balance_ratio(balance_ratio), chunk_size_B(chunk_size_B), a2a_chunk_parent(nullptr), a2a_chunk_rem_size(nullptr), used_layers(used_layers) {
  assert(sched != nullptr);
  int nchunks = sched->src_dest_pairs.size();
  std::map<std::pair<int, int>, int> pair_sizes;
  for (auto &[chunk_id, pairs] : sched->src_dest_pairs) {
    auto &[srcs, dsts] = pairs;
    int src_size = srcs.size(), dst_size = dsts.size();
    pair_sizes[{src_size, dst_size}]++;
  }
  hash = {nchunks, pair_sizes};
}

void _solveSubgroup(int layer_id, int group_id, Solver::ChunkSize chunk_size_B, int spilt_chunks, double balance_ratio, intra_group_sched &group_sched, std::map<int, std::vector<int>> a2a_chunk_parent, std::map<int, std::vector<int>> a2a_chunk_rem_size, std::shared_ptr<Simulation::Topology> topo, std::shared_ptr<Solver::MIPSolver> solver, std::shared_ptr<Solver::EventSolution> &ret, bool a2a) {
  auto _solver = solver->clone();
  ret = solveSubgroup(layer_id, group_id, chunk_size_B, spilt_chunks, balance_ratio, group_sched, a2a_chunk_parent, a2a_chunk_rem_size, topo, _solver, a2a);
  assert(ret != nullptr);
  LOOOG(ALGO, TRACE, "Solved subgroup Demand @ layer %d, group %d, split %d, balance %.2lf, chunk %llu", layer_id, group_id, spilt_chunks, balance_ratio, chunk_size_B);
}

DemandSolveReport addAndSolveSubDemandsMultiThread(std::vector<SketchSubDemand> &sub_demands, std::vector<std::pair<std::shared_ptr<Solver::EventSolution>, demand_map_t>> &result, std::shared_ptr<Simulation::Topology> topo, std::shared_ptr<Solver::MIPSolver> solver, bool a2a) {
  DemandSolveReport ret;
  Clock_ clock;
  // group by layer, split and hash first
  result.resize(sub_demands.size());
  std::fill(result.begin(), result.end(), std::make_pair(nullptr, demand_map_t()));
  std::map<solver_param_key_t, std::map<subdemand_hash_t, std::vector<int>>> layer_hash_map;
  // isomorphic subdemands should has the same {sub_demand.layer_id, sub_demand.split_chunks, sub_demand.balance_ratio, sub_demand.chunk_size_B, hash} pair
  // The hash here is calculated as (number of chunks, distribution of size of (srcs, dsts) pairs for every chunk)
  // A valid chunk and GPU mapping can only exist if two subdemands' hash is the same
  // For alltoall, we need to further check the routing info between srcs and dsts, this is done later
  for (int i = 0; i < sub_demands.size(); i++) {
    auto &sub_demand = sub_demands[i];
    layer_hash_map[{sub_demand.layer_id, sub_demand.split_chunks, sub_demand.balance_ratio, sub_demand.chunk_size_B}][sub_demand.hash].push_back(i);
  }

  // We record relevant variables for each key-hash pair here, as they are solved in parallel
  std::map<std::pair<solver_param_key_t, subdemand_hash_t>, std::vector<std::pair<int, intra_group_sched*>>> key_input_groups;
  std::map<std::pair<solver_param_key_t, subdemand_hash_t>, std::vector<int>> key_local_to_global_id;
  std::map<std::pair<solver_param_key_t, subdemand_hash_t>, std::vector<std::pair<int, demand_map_t>>> key_merged_res;
  std::map<std::pair<solver_param_key_t, subdemand_hash_t>, std::map<int, std::shared_ptr<Solver::EventSolution>>> key_solution;
  std::map<std::pair<solver_param_key_t, subdemand_hash_t>, std::set<int>> key_used_layers;
  std::map<intra_group_sched*, std::map<int, std::vector<int>>*> sched_a2a_chunk_parent, sched_a2a_chunk_rem_size;

  // generate input groups for every key
  for (auto &[key, hash_map] : layer_hash_map) {
    auto [layer_id, split_chunks, balance_ratio, chunk_size_B] = key;
    for (auto &[hash, ids] : hash_map) {
      // ids having the same hash
      auto &input_groups = key_input_groups[{key, hash}];
      auto &local_to_global_id = key_local_to_global_id[{key, hash}];
      for (auto id : ids) {
        assert(result[id].first == nullptr);
        if (result[id].first == nullptr) {
          input_groups.push_back({sub_demands[id].group_id, sub_demands[id].sched});
          sched_a2a_chunk_parent[sub_demands[id].sched] = sub_demands[id].a2a_chunk_parent;
          sched_a2a_chunk_rem_size[sub_demands[id].sched] = sub_demands[id].a2a_chunk_rem_size;
          local_to_global_id.push_back(id);
        }
        if (key_used_layers.count({key, hash})) {
          bool same_used_layers = key_used_layers[{key, hash}] == sub_demands[id].used_layers;
          if (!same_used_layers) {
            LOOOG(ALGO, INFO, "Clash for key (%d, %d, %lf, %llu) #%d", layer_id, split_chunks, balance_ratio, chunk_size_B, id);
            LOOOG(ALGO, INFO, "key_used_layers: %s, used_layers: %s", key_used_layers[{key, hash}], sub_demands[id].used_layers);
          }
          assert(same_used_layers);
        }
        key_used_layers[{key, hash}] = sub_demands[id].used_layers;
      }
      for (auto input : input_groups) {
        LOOOG(ALGO, DEBUG, "Input Group: %d, %s", input.first, input.second->toStr().c_str());
      }
    }
  }

  // solve isomorphism for each key
  ctpl::thread_pool pool(PARALLEL_THREAD_NUM), shared_pool(PARALLEL_THREAD_NUM);
  for (auto &[_key, input_groups] : key_input_groups) {
    auto &[key, hash] = _key;
    auto &[layer_id, split_chunks, balance_ratio, chunk_size_B] = key;
    LOOOG(ALGO, INFO, "Solving isomorphism for %d objects", input_groups.size());
    ret.ndemands += input_groups.size();
    auto &merged_res = key_merged_res[_key];
    auto &used_layers = key_used_layers[_key];
    pool.push([&](int id) {
      merge_isomorphic_demand_groups(std::ref(input_groups), topo, layer_id, std::ref(merged_res), std::ref(used_layers), std::ref(shared_pool), a2a, std::ref(sched_a2a_chunk_parent), std::ref(sched_a2a_chunk_rem_size));
    });  
  }
  pool.stop(true);
  assert(pool.size() == 0);

  assert(PERFMODEL_POOL_SIZE == PARALLEL_THREAD_NUM);
  ctpl::thread_pool pool2(PARALLEL_THREAD_NUM);
  for (auto &[_key, input_groups] : key_input_groups) {
    auto &[key, hash] = _key;
    auto &[layer_id, split_chunks, balance_ratio, chunk_size_B] = key;
    auto &local_to_global_id = key_local_to_global_id[_key];
    auto &merged_res = key_merged_res[_key];
    // Solve for the representatives first
    for (int i = 0; i < merged_res.size(); i++) {
      auto &[rep, dmap] = merged_res[i];
      if (rep == -1) {
        LOOOG(ALGO, DEBUG, "Solving for key (%d, %d, %lf, %llu) #%d", layer_id, split_chunks, balance_ratio, chunk_size_B, i);
        // Solve for the representative
        auto &[group_id, group_sched] = input_groups[i];
        ret.nsolves++;
        auto &solve_res = key_solution[_key][i];
        auto &sub_demand = sub_demands[local_to_global_id[i]];
        // Actual subdemand solving with TE-CCL MILP model, can replace with other models
        pool2.push([&](int id) {
          _solveSubgroup(layer_id, group_id, chunk_size_B, split_chunks, balance_ratio, std::ref(*group_sched), std::ref(*sub_demand.a2a_chunk_parent), std::ref(*sub_demand.a2a_chunk_rem_size), topo, solver, std::ref(solve_res), a2a);
        });  
      }
    }
  }
  pool2.stop(true);
  assert(pool2.size() == 0);

  LOOOG(ALGO, TRACE, "Merging results");
  // return representatives and to_be_mapped schedules
  for (auto &[_key, input_groups] : key_input_groups) {
    auto &[key, hash] = _key;
    auto [layer_id, split_chunks, balance_ratio, chunk_size_B] = key;
    auto &local_to_global_id = key_local_to_global_id[_key];
    auto &merged_res = key_merged_res[_key];
    for (int i = 0; i < merged_res.size(); i++) {
      auto &[rep, dmap] = merged_res[i];
      if (rep == -1) {
        LOOOG(ALGO, DEBUG, "Mapping for key (%d, %d, %lf, %llu) #%d", layer_id, split_chunks, balance_ratio, chunk_size_B, i);
        auto &solve_res = key_solution[_key][i];
        assert(solve_res != nullptr);
        result[local_to_global_id[i]] = {solve_res, demand_map_t()};
      }
    }
    // Map the results
    for (int i = 0; i < merged_res.size(); i++) {
      auto &[rep, dmap] = merged_res[i];
      if (rep != -1) {
        assert(result[local_to_global_id[rep]].first != nullptr);
        result[local_to_global_id[i]] = {result[local_to_global_id[rep]].first, dmap};
      }
    }
  }

  LOOOG(ALGO, TRACE, "Finished merging results");
  return ret;
}

std::shared_ptr<Solver::EventSolution> mapSchedule(std::shared_ptr<Solver::EventSolution> orig, demand_map_t &dmap, bool a2a) {
  assert(dmap.checkValid());
  auto new_res = std::make_shared<Solver::EventSolution>();
  for (auto &[src_c, evs] : orig->events) {
    assert(dmap.chunk_map.count(src_c.first));
    Solver::SrcChunk new_chunk_id;
    if (a2a){
      assert(dmap.gpu_map.count(src_c.second));
      new_chunk_id = {dmap.chunk_map[src_c.first], dmap.gpu_map[src_c.second]};  
    } else {
      new_chunk_id = {dmap.chunk_map[src_c.first], src_c.second};
    }
    for (auto &ev : evs) {
      auto new_ev = ev;
      assert(dmap.gpu_map.count(ev.src_gpu) && dmap.gpu_map.count(ev.dst_gpu));
      new_ev.src_gpu = dmap.gpu_map[ev.src_gpu];
      new_ev.dst_gpu = dmap.gpu_map[ev.dst_gpu];
      new_res->events[new_chunk_id].push_back(new_ev);
    }
  }
  return new_res;
}

std::shared_ptr<Solver::EventSolution> reconstructEventa2a(std::shared_ptr<Solver::EventSolution> full_sched, int ngpus) {
  std::map<Solver::SrcChunk, std::map<int, std::map<int, double>>> chunk_event_cnt;
  std::map<Solver::SrcChunk, std::map<int, std::map<int, uint64_t>>> chunk_min_ep;
  auto processed_sched = std::make_shared<Solver::EventSolution>();
  auto &events = full_sched->events;
  for (auto &[chunk, evs] : events) {
    for (auto &ev : evs) {
      chunk_event_cnt[chunk][ev.dst_gpu][ev.src_gpu] += ev.amount;
      if (!chunk_min_ep[chunk][ev.dst_gpu].count(ev.src_gpu)) {
        chunk_min_ep[chunk][ev.dst_gpu][ev.src_gpu] = ev.epoch;
      } else {
        chunk_min_ep[chunk][ev.dst_gpu][ev.src_gpu] = std::min(chunk_min_ep[chunk][ev.dst_gpu][ev.src_gpu], ev.epoch);
      }
    }
  }
  for (int src_c = 0; src_c < ngpus; ++src_c) {
    for (int dst_c = 0; dst_c < ngpus; ++dst_c) {
      if (src_c == dst_c) {
        continue;
      }
      LOOOG(ALGO, TRACE, "Find a2a path for %d -> %d", src_c, dst_c);
      int cur_prev = dst_c;
      Solver::SrcChunk real_chunk = {src_c, dst_c};
      while (cur_prev != src_c) {
        Solver::SrcChunk chunk = {src_c, cur_prev};
        auto &sends = chunk_event_cnt[chunk][cur_prev];
        double rem = 1.0;
        LOOOG(ALGO, TRACE, "find path %d -> %d", src_c, cur_prev);
        std::vector<int> srcs;
        for (auto &[src, amount] : sends) {
          LOOOG(ALGO, TRACE, "Amount %d -> %d: %.32lf", src, cur_prev, amount);
          if (amount > 0) {
            assert(amount > 1e-1);
            srcs.push_back(src);
          }
        }
        std::sort(srcs.begin(), srcs.end(), [&](int a, int b) {
          return chunk_min_ep[chunk][cur_prev][a] < chunk_min_ep[chunk][cur_prev][b];
        });
        assert(!srcs.empty());
        for (auto src : srcs) {
          auto &amount = chunk_event_cnt[chunk][cur_prev][src];
          // LOOOG(ALGO, TRACE, "Amount %d -> %d: %.32lf, rem %.32lf", src, cur_prev, amount, rem);
          if (amount - rem >= -1e-1) {
            amount -= rem;
            assert(amount >= -1e-1);
            processed_sched->addLPEvent(real_chunk, src, cur_prev, chunk_min_ep[chunk][cur_prev][src], 1.0);
            rem = 0; cur_prev = src;
            break;
          } else {
            rem -= amount;
            amount = 0;
            assert(rem >= 1e-1);
          }
        }
        assert(rem == 0);
      }
    }
  }
  return processed_sched;
}

void processFinalSchedule(std::shared_ptr<solved_schedule_t> &ret, std::shared_ptr<sketch_solve_workload_t> &wr, std::vector<int> &final_order, std::shared_ptr<Simulation::PerfModel> &perf_model, std::shared_ptr<Simulation::Topology> &topo, const std::vector<std::pair<std::shared_ptr<Solver::EventSolution>, demand_map_t>> &sched_result, bool a2a) {
  auto chunk_size_B = wr->chunk_size_B;
  auto split_chunks = wr->split_chunks;
  uint64_t cur_time = 0, nxt_time = 0;
  int ngpus = wr->sketch->ngpus;
  auto full_sched = std::make_shared<Solver::EventSolution>();
  // concat the schedules in topological order
  for (auto i : final_order) {
    auto &res = sched_result[i].first;
    for (auto &[src_c, evs] : res->events) {
      for (auto ev : evs) {
        ev.epoch += cur_time;
        full_sched->events[src_c].push_back(ev);
        nxt_time = std::max(nxt_time, ev.epoch);
      }
    }
    cur_time = nxt_time + 1;
  }
  Simulation::SimConfig sim_config;
  if (a2a) {
    auto processed_sched = reconstructEventa2a(full_sched, ngpus);
    for (int chunk_id = 0; chunk_id < ngpus; ++chunk_id) {
      for (int i = 0; i < ngpus; ++i) {
        Solver::SrcChunk chunk = {chunk_id, i};
        processed_sched->setChunkSize(chunk, chunk_size_B);
        int src_dev = topo->getNpuDevice(chunk_id)->get_id();
        processed_sched->addInitDev(chunk, src_dev);
      }
    }  
    sim_config.sol = processed_sched;
  } else {
    for (auto &node : wr->sketch->nodes) {
      // insert root chunks into the result as sources
      if (node.deps.size() == 0) {
        for (auto &[group_id, group_sched] : node.group_scheds) {
          for (auto &[chunk_id, pairs] : group_sched.src_dest_pairs) {
            for (int i = 0; i < split_chunks; ++i) {
              Solver::SrcChunk chunk = {chunk_id, i};
              auto &[srcs, dsts] = pairs;
              full_sched->setChunkSize(chunk, chunk_size_B / split_chunks);
              for (auto src : srcs) {
                int src_dev = topo->getNpuDevice(src)->get_id();
                full_sched->addInitDev(chunk, src_dev);
              }
            }
          }
        }
      }
    }      
    sim_config.sol = full_sched;
  }
  LOOOG(ALGO, DEBUG, "Solved Epoch Schedule: %s", full_sched->toStr().c_str());
  auto schedule = perf_model->runAnalysis(sim_config);
  ret->sim_time_ms = 0;
  ret->schedule = schedule;
  ret->chunk_size_B = wr->chunk_size_B;
}

std::vector<std::shared_ptr<solved_schedule_t>> Algorithm::sketchesSolve(std::vector<std::shared_ptr<sketch_solve_workload_t>> wrs, bool a2a) {
  auto topo = perf_model->getTopo();
  std::vector<std::shared_ptr<solved_schedule_t>> rets(wrs.size());
  std::vector<std::vector<int>> wr_final_order(wrs.size());
  std::vector<SketchSubDemand> sub_demands;

  // Split every sketch into subdemands
  for (auto wr : wrs) {
    int id = wr->id;
    auto sketch = wr->sketch;
    auto balance_ratio = wr->balance_ratio;
    auto chunk_size_B = wr->chunk_size_B;
    auto split_chunks = wr->split_chunks;
    auto &a2a_chunk_parent = sketch->a2a_chunk_parent;
    auto &a2a_chunk_rem_size = sketch->a2a_chunk_rem_size;
    assert(chunk_size_B % split_chunks == 0);
    std::map<int, int> in_deg;
    std::queue<int> q;
    for (auto &node : sketch->nodes) {
      in_deg[node.id] = node.deps.size();
      if (in_deg[node.id] == 0) {
        q.push(node.id);
      }
    }
    auto &final_order = wr_final_order[id];
    // Find the topological order of the full schedule, so that we know how to combine the solved schedules of subdemands in every node
    while (!q.empty()) {
      int id = q.front(); q.pop();
      auto &node = sketch->nodes[id];
      int layer_id = node.layer;
      for (auto &[group_id, group_sched] : node.group_scheds) {
        // Every group in the node(with all the source chunks) is a subdemand
        sub_demands.emplace_back(layer_id, group_id, &group_sched, split_chunks, balance_ratio, chunk_size_B, wr->sketch->used_layers);
        sub_demands.back().a2a_chunk_parent = &a2a_chunk_parent;
        sub_demands.back().a2a_chunk_rem_size = &a2a_chunk_rem_size;
        final_order.push_back(sub_demands.size() - 1);
      }
      for (auto nxt : node.next) {
        in_deg[nxt]--;
        if (in_deg[nxt] == 0) {
          q.push(nxt);
        }
      }
    }
  }

  // (pointer to isomorphic schedule representative, the mapping)
  std::vector<std::pair<std::shared_ptr<Solver::EventSolution>, demand_map_t>> sched_result;
  LOOOG(ALGO, INFO, "Total subdemands: %d", sub_demands.size());
  DemandSolveReport rep;
  rep = addAndSolveSubDemandsMultiThread(sub_demands, sched_result, topo, std::dynamic_pointer_cast<Solver::MIPSolver>(solver), a2a);

  for (int i = 0; i < sched_result.size(); i++) {
    auto &[res, dmap] = sched_result[i];
    assert(res != nullptr);
    assert(dmap.checkValid());
    if (dmap.gpu_map.empty() && dmap.chunk_map.empty()) {
      // representative
      LOOOG(ALGO, DEBUG, "Representative %d: %s", i, res->toStr().c_str());
      continue;
    }
    LOOOG(ALGO, DEBUG, "Original %d: %s", i, res->toStr().c_str());
    // Directly map the schedules here
    auto new_res = mapSchedule(res, dmap, a2a);
    LOOOG(ALGO, DEBUG, "Mapped %d: %s", i, new_res->toStr().c_str());
    sched_result[i].first = new_res;
  }

  ctpl::thread_pool pool(PERFMODEL_POOL_SIZE);
  // Combine all the solved subdemand schedules into a complete schedulem then generate the final schedule via simulation
  for (auto &wr : wrs) {
    int id = wr->id;
    auto &ret_p = rets[id];
    ret_p = std::make_shared<solved_schedule_t>();
    ret_p->parse_time_ms = rep.iso_time_ms;
    ret_p->solver_time_ms = rep.solver_time_ms;
    ret_p->ndemands = rep.ndemands;
    ret_p->nsolves = rep.nsolves;
    auto &wr_final_order_p = wr_final_order[id];
    pool.push([&](int th_id) {
      assert(th_id < PERFMODEL_POOL_SIZE);
      processFinalSchedule(std::ref(ret_p), std::ref(wr), std::ref(wr_final_order_p), perf_model_pool[th_id], std::ref(topo), std::ref(sched_result), a2a);
    });
  }
  pool.stop(true);
  return rets;
}
