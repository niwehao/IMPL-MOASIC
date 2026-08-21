// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/CommGraph.h"
#include "PerfModel/PerfModel.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include <cassert>
#include <memory>
#include <thread>
#include "common/ctpl.h"

using BandwithUsage = std::map<int, double>;
// layer -> bandwidth usage in terms of bytes communicated
// Note that we didn't consider overlapping links between layers/interleaving link usage between steps here
// therefore more complex combination strategies can be implemented
BandwithUsage calculateBandwidthUsage(std::shared_ptr<full_sched_graph_t> sched, bool a2a) {
  std::map<int, double> bandwidth_usage;
  for (auto &node : sched->nodes) {
    int layer_id = node.layer;
    for (auto &[group_id, sched_] : node.group_scheds) {
      for (auto &[chunk_id, src_dst] : sched_.src_dest_pairs) {
        auto &[srcs, dsts] = src_dst;
        if (!a2a) {
          bandwidth_usage[layer_id] += dsts.size();
        } else {
          for (auto src : srcs) {
            bandwidth_usage[layer_id] += sched->a2a_chunk_rem_size[chunk_id][src];
          }
        }
      }
    }
  }
  return bandwidth_usage;
}

std::pair<double, double> solveCombRatio(std::vector<double> x, std::vector<double> y, std::vector<double> l) {
  assert(x.size() == 2 && y.size() == 2);
  // a + b = 1, (a * x[0] + b * y[0]) / (a * x[1] + b * y[1]) = l[0] / l[1]
  LOOOG(ALGO, TRACE, "x = {%.2lf, %.2lf}, y = {%.2lf, %.2lf}, l = {%.2lf, %.2lf}", x[0], x[1], y[0], y[1], l[0], l[1]);
  double div = ((x[0] - y[0]) - l[0] / l[1] * (x[1] - y[1]));
  if (div == 0) {
    return {-1, -1};
  }
  double a = (l[0] / l[1] * y[1] - y[0]) / div;
  double b = 1 - a;
  if (a < 0 || a > 1) {
    return {-1, -1};
  }
  assert(fabs(
    (a * x[0] + b * y[0]) / (a * x[1] + b * y[1]) - (l[0] / l[1])
  ) < 1e-6);
  return {a, b};
}

std::vector<std::shared_ptr<combined_sketch_t>> Algorithm::combineSketches(std::vector<std::shared_ptr<full_sched_graph_t>> &sketches, bool a2a) {
  std::vector<std::shared_ptr<combined_sketch_t>> ret;
  auto topo = perf_model->getTopo();
  auto &topo_info = perf_model->getTopo()->topo_info;
  auto layer_info = ExtractedLayerInfo(topo_info, std::set<int>());
  std::vector<BandwithUsage> bandwidth_usages;
  // Single Sketches
  for (auto &sched : sketches) {
    bandwidth_usages.push_back(calculateBandwidthUsage(sched, a2a));
    LOOOG(ALGO, TRACE, "INFO: %s", sched->toStr().c_str());
    for (auto &[layer_id, usage] : bandwidth_usages.back()) {
      LOOOG(ALGO, TRACE, "Layer %d: %.2lf", layer_id, usage);
    }
    ret.push_back(std::make_shared<combined_sketch_t>());
    ret.back()->addSketch(sched, 1);
  }
  // Two Sketch Combination, We ignore the case of more sketches for now
  for (int i = 0; i < sketches.size(); i++) {
    for (int j = i + 1; j < sketches.size(); j++) {
      auto &sched_i = sketches[i];
      auto &sched_j = sketches[j];
      auto &usage_i = bandwidth_usages[i];
      auto &usage_j = bandwidth_usages[j];
      LOOOG(ALGO, INFO, "Trying to Combine %d and %d", i, j);
      LOOOG(ALGO, INFO, "Usage %d:", i);
      for (auto &[layer_id, usage] : usage_i) {
        LOOOG(ALGO, INFO, "Layer %d: %.2lf", layer_id, usage);
      }
      LOOOG(ALGO, INFO, "Usage %d:", j);
      for (auto &[layer_id, usage] : usage_j) {
        LOOOG(ALGO, INFO, "Layer %d: %.2lf", layer_id, usage);
      }
      std::vector<double> x, y;
      std::vector<int> layer_ids;
      std::set<int> layer_set;
      for (auto &[layer_id, usage] : usage_i) {
        layer_set.insert(layer_id);
      }
      for (auto &[layer_id, usage] : usage_j) {
        layer_set.insert(layer_id);
      }
      for (auto &layer_id : layer_set) {
        x.push_back(usage_i.count(layer_id) ? usage_i[layer_id] : 0);
        y.push_back(usage_j.count(layer_id) ? usage_j[layer_id] : 0);
        layer_ids.push_back(layer_id);
      }
      // combine if there are at least 2 layers used
      // We only consider the ratio between intra and inter machine now
      // if there are more than 2 layers, we combine them into 2 instead of actually solving the whole thing
      // Consider the links of higher layers are actually overlapped in Clos, this is actually a good approximation
      if (x.size() > 2) {
        for (int i = 2; i < x.size(); i++) {
          x[1] += x[i], y[1] += y[i];
        }
        x = {x[0], x[1]}, y = {y[0], y[1]}, layer_ids = {layer_ids[0], layer_ids[1]};
      }
      if (x.size() != 2) {
        continue;
      }
      int l0 = layer_ids[0], l1 = layer_ids[1];
      // Note this calculation may not be correct for non-switched topologies
      auto bw0 = topo_info.get_layer_avg_bandwidth(l0), bw1 = topo_info.get_layer_avg_bandwidth(l1);
      LOOOG(ALGO, INFO, "Real ratio: Layer %d: %.2lf, Layer %d: %.2lf", l0, bw0, l1, bw1);
      std::vector<double> l = {bw0, bw1};
      auto [a, b] = solveCombRatio(x, y, l);
      LOOOG(ALGO, TRACE, "Ratio: %.2lf %.2lf", a, b);
      if (a > 0) {
        LOOOG(ALGO, INFO, "Combining %d and %d with ratio %.2lf %.2lf", i, j, a, b);
        ret.push_back(std::make_shared<combined_sketch_t>());
        ret.back()->addSketch(sched_i, a);
        ret.back()->addSketch(sched_j, b);
      }
    }
  }
  return ret;
}

std::vector<int> splitToRatio(int tot, const std::vector<double> &ratios) {
  std::vector<int> ret;
  double sum = 0;
  for (auto ratio : ratios) {
    sum += ratio;
  }
  int rem = tot;
  for (auto ratio : ratios) {
    int val = tot * ratio / sum;
    ret.push_back(val);
    rem -= val;
  }
  std::vector<std::pair<int, double>> idx_ratio;
  for (int i = 0; i < ratios.size(); i++) {
    idx_ratio.push_back({i, ratios[i]});
  }
  std::sort(idx_ratio.begin(), idx_ratio.end(), [](auto &a, auto &b) {
    return a.second > b.second;
  });
  for (int i = 0; i < rem; i++) {
    ret[idx_ratio[i].first]++;
  }
  int sum_ret = 0;
  for (auto val : ret) {
    sum_ret += val;
  }
  assert(sum_ret == tot);
  return ret;
}

void generateCombinedSketchAlgo(std::shared_ptr<combined_sketch_t> &sketch, std::shared_ptr<generated_algorithm_t> &algo, std::map<std::shared_ptr<combined_sketch_t>, std::vector<int>> &combined_sketch_to_workloads, std::vector<std::shared_ptr<solved_schedule_t>> &results, std::shared_ptr<Simulation::PerfModel> perf_model, double balance_ratio, uint64_t chunk_size_B) {
  if (!combined_sketch_to_workloads.count(sketch)) {
    algo = nullptr;
    return;
  }
  assert(combined_sketch_to_workloads[sketch].size() == sketch->sketches.size());
  algo = std::make_shared<generated_algorithm_t>();
  algo->sketch = sketch;
  algo->balance_ratio = balance_ratio;
  algo->chunk_size_B = chunk_size_B;
  for (int i = 0; i < sketch->sketches.size(); i++) {
    auto wr_id = combined_sketch_to_workloads[sketch][i];
    auto res = results[wr_id];
    algo->schedules.push_back(res);
  }
  LOOOG(ALGO, TRACE, "Combining Solved Schedules");
  auto sim_config = Simulation::SimConfig();
  sim_config.sol = std::make_shared<Solver::EventSolution>();
  auto &sol = sim_config.sol;
  int idx_start = 0;
  for (int i = 0; i < sketch->sketches.size(); i++) {
    LOOOG(ALGO, TRACE, "Processing Schedule %d/%d", i, sketch->sketches.size());
    auto &sched = algo->schedules[i]->schedule->schedule;
    assert(sched != nullptr);
    int max_idx = 0;
    for (auto &[chunk, events] : sched->events) {
      assert(sched->init_devs.count(chunk) && sched->chunk_sizes.count(chunk));
      auto [src, idx] = chunk;
      Solver::SrcChunk new_chunk = {src, idx_start + idx};
      max_idx = std::max(max_idx, idx_start + idx);
      for (auto &event : events) {
        sol->events[new_chunk].push_back(event);
      }
      for (auto &dev : sched->init_devs[chunk]) {
        sol->init_devs[new_chunk].insert(dev);
      }
      sol->chunk_sizes[new_chunk] = sched->chunk_sizes[chunk];
    }
    idx_start = max_idx + 1;
  }
  auto combined_sched = perf_model->runAnalysis(sim_config);
  algo->final_schedule = combined_sched;
}

/*
TE-CCL supports splitting chunks into uniform-sized ones for better solution during modeling, so we have a parameter here
However it's not used in the current implementation as sketch combination covers that
Input contains multiple sketches to be break down into subdemands, grouped by isomorphism, and then solved in parallel
We record simulation time, and the rest is considered solving time
*/
std::vector<std::shared_ptr<generated_algorithm_t>> Algorithm::combinedSketchesSolve(std::vector<std::shared_ptr<combined_sketch_t>> sketches, double balance_ratio, uint64_t chunk_size_B, int split_chunks, bool a2a, double &comb_sim_time) {
  std::vector<std::shared_ptr<generated_algorithm_t>> generated_algos;
  // split into individual solving workloads
  std::vector<std::shared_ptr<sketch_solve_workload_t>> workloads;
  std::map<std::shared_ptr<combined_sketch_t>, std::vector<int>> combined_sketch_to_workloads;
  int nwrs = 0;
  for (auto sketch : sketches) {
    auto &ratios = sketch->chunk_size_ratio;
    assert(ratios.size() == sketch->sketches.size());
    std::vector<uint64_t> chunk_sizes;
    // do not split if the chunk size is too small
    if (chunk_size_B <= SLICE_SIZE_B && ratios.size() > 1) {
      continue;
    }
    int nslices = chunk_size_B / SLICE_SIZE_B;
    nslices = std::max(nslices, 1);
    if (chunk_size_B > SLICE_SIZE_B && chunk_size_B % SLICE_SIZE_B != 0) {
      LOOOG(ALGO, INFO, "WARNING: Chunk size %llu is not a multiple of slice size %d, Wrong input?", chunk_size_B, SLICE_SIZE_B);
      // assert(false);
      nslices++;
    }
    bool invalid = false;
    // round up the chunk sizes in unit of slices
    auto slice_sizes = splitToRatio(nslices, ratios);
    for (auto sz : slice_sizes) {
      if (sz == 0) {
        invalid = true;
      }
    }
    if (invalid) {
      continue;
    }
    LOOOG(ALGO, TRACE, "Chunk Size: %llu, Slice Count: %d", chunk_size_B, nslices);
    Solver::ChunkSize rem_size = chunk_size_B;
    for (int i = 0; i < sketch->sketches.size(); i++) {
      Solver::ChunkSize sketch_chunk_size = slice_sizes[i] * std::min(chunk_size_B, SLICE_SIZE_B);
      sketch_chunk_size = std::min(sketch_chunk_size, rem_size);
      rem_size -= sketch_chunk_size;
      chunk_sizes.push_back(sketch_chunk_size);
      assert(chunk_sizes.back() != 0);
      LOG(ALGO, TRACE, "Sketch %d: %llu Bytes, ratio %.2lf", i, chunk_sizes[i], ratios[i]);
      auto workload = std::make_shared<sketch_solve_workload_t>(sketch->sketches[i], balance_ratio, sketch_chunk_size, split_chunks, nwrs++);
      workloads.push_back(workload);
      combined_sketch_to_workloads[sketch].push_back(workloads.size() - 1);
    }
    assert(rem_size == 0);
  }

  LOOOG(ALGO, INFO, "Solving %d Sketch Workloads", workloads.size());

  auto results = sketchesSolve(workloads, a2a);

  LOOOG(ALGO, INFO, "Solved %d Sketch Workloads, Combining into final shedule...", results.size());

  // Combine the results into final schedules and do simulation
  ctpl::thread_pool pool(PERFMODEL_POOL_SIZE);
  generated_algos.resize(sketches.size());
  Clock_ clk; clk.tic();
  int i = 0;
  for (auto &sketch : sketches) {
    auto &generated_algo_p = generated_algos[i];
    pool.push([&](int id) {
      assert(id < PERFMODEL_POOL_SIZE);
      generateCombinedSketchAlgo(std::ref(sketch), std::ref(generated_algo_p), std::ref(combined_sketch_to_workloads), std::ref(results), perf_model_pool[id], balance_ratio, chunk_size_B);
    });
    i += 1;
  }
  pool.stop(true);
  generated_algos.erase(std::remove(generated_algos.begin(), generated_algos.end(), nullptr), generated_algos.end());
  comb_sim_time = clk.tic();
  LOOOG(ALGO, INFO, "CombinedSketchSolve Finished Generated %d Algorithms", generated_algos.size());
  return generated_algos;
}
