// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/Algorithm.h"
#include "Sketch/CommGraph.h"
#include "Sketch/common.h"
#include "Solver/common.h"
#include <cassert>

using nlohmann::json;

json intra_group_sched::to_json() {
  json ret;
  ret["src_dest_pairs"] = json::array();
  for (auto &[chunk_id_, sends] : src_dest_pairs) {
    ret["src_dest_pairs"].push_back(json::object());
    auto &obj = ret["src_dest_pairs"].back();
    obj["chunk_id"] = format("%d", chunk_id_);
    auto &[srcs, dsts] = sends;
    obj["srcs"] = srcs;
    obj["dsts"] = dsts;
  }
  return ret;
}

json full_sched_graph_node_t::to_json() {
  json ret;
  ret["id"] = id;
  ret["step"] = step;
  ret["layer"] = layer;
  ret["group_scheds"] = json::object();
  for (auto &[group_id, group_sched] : group_scheds) {
    ret["group_scheds"][format("Group#%d", group_id)] = group_sched.to_json();
  }
  ret["next"] = next;
  ret["deps"] = deps;
  return ret;
}

json sched_graph_node_t::to_json() {
  json ret;
  ret["id"] = id;
  ret["step"] = step;
  ret["layer"] = layer;
  ret["group"] = group;
  ret["src_dest_pair"] = json::object();
  auto &[srcs, dsts] = src_dest_pair;
  ret["src_dest_pair"]["srcs"] = srcs;
  ret["src_dest_pair"]["dsts"] = dsts;
  ret["deps"] = deps;
  ret["next"] = next;
  return ret;
}

json sched_graph_t::to_json() {
  json ret;
  ret["ngpus"] = ngpus;
  ret["src_gpu"] = src_gpu;
  ret["nodes"] = json::array();
  for (int i = 0; i < nodes.size(); ++i) {
    ret["nodes"].push_back(nodes[i].to_json());
  }
  return ret;
}

json full_sched_graph_t::to_json() {
  json ret;
  ret["ngpus"] = ngpus;
  ret["nodes"] = json::array();
  assert(src_graph != nullptr);
  ret["src_graph"] = src_graph->to_json();
  for (int i = 0; i < nodes.size(); ++i) {
    ret["nodes"].push_back(nodes[i].to_json());
  }
  return ret;
}

json combined_sketch_t::to_json() {
  json ret;
  ret["sketches"] = json::array();
  for (int i = 0; i < sketches.size(); ++i) {
    auto ratio = chunk_size_ratio[i];
    ret["sketches"].push_back(sketches[i]->to_json());
    ret["sketches"].back()["chunk_size_ratio"] = ratio;
  }
  return ret;
}

json solved_schedule_t::to_json() {
  json ret;
  ret["chunk_size_B"] = chunk_size_B;
  ret["parse_time_ms"] = parse_time_ms;
  ret["solver_time_ms"] = solver_time_ms;
  ret["sim_time_ms"] = sim_time_ms;
  ret["ndemands"] = ndemands;
  ret["nsolves"] = nsolves;
  ret["schedule"] = schedule->to_json();
  return ret;
}

json Solver::ScheduleEvent::to_json() {
  json ret;
  ret["src_gpu"] = src_gpu;
  ret["dst_gpu"] = dst_gpu;
  ret["epoch"] = epoch;
  ret["amount"] = amount;
  ret["copy"] = copy;
  ret["reduce"] = reduce;
  ret["layer_used"] = layer_used;
  return ret;
}

json Solver::EventSolution::to_json() {
  json ret;
  ret["init_devs"] = init_devs;
  ret["chunk_sizes"] = chunk_sizes;
  ret["Events"] = json::array();
  for (auto &[src_c, evs] : events) {
    ret["Events"].push_back(json::object());
    auto &e_src = ret["Events"].back();
    e_src["src_chunk"] = format("(%d, %d)", std::get<0>(src_c), std::get<1>(src_c));
    e_src["sends"] = json::array();
    for (auto &ev : evs) {
      e_src["sends"].push_back(ev.to_json());
    }
  }
  return ret;
}

std::shared_ptr<Solver::EventSolution> Solver::EventSolution::from_json(nlohmann::json &j) {
    // Create a shared pointer to a new EventSolution
    auto solution = std::make_shared<EventSolution>();
    // Extract init_devs
    solution->init_devs = j.at("init_devs").get<std::map<SrcChunk, std::set<int>>>();
    // Extract chunk_sizes
    solution->chunk_sizes = j.at("chunk_sizes").get<std::map<SrcChunk, ChunkSize>>();
    // Extract events
    for (const auto& e_src : j.at("Events")) {
        std::string src_chunk_str = e_src.at("src_chunk").get<std::string>();
        int chunk_1, chunk_2;
        // Assuming src_chunk is formatted as "(chunk_1, chunk_2)"
        sscanf(src_chunk_str.c_str(), "(%d, %d)", &chunk_1, &chunk_2);
        std::pair<int, int> src_chunk = std::make_pair(chunk_1, chunk_2);
        auto &events = solution->events[src_chunk];
        // Extract sends
        for (const auto& send : e_src.at("sends")) {
            int src_gpu = send.at("src_gpu").get<int>();
            int dst_gpu = send.at("dst_gpu").get<int>();
            EventTime epoch = send.at("epoch").get<EventTime>();
            bool copy = send.at("copy").get<bool>();
            bool reduce = send.at("reduce").get<bool>();
            double amount = send.value("amount", 1.0);
            int layer_used = send.value("layer_used", -1);
            events.emplace_back(src_gpu, dst_gpu, epoch, amount, copy, reduce);
            events.back().layer_used = layer_used;
        }
    }

    // Return the shared pointer
    return solution;
}

json Solver::SimResult::to_json() {
  json ret;
  assert(schedule != nullptr);
  ret["Schedule"] = schedule->to_json();
  ret["ReceivedChunks"] = received_chunks;
  ret["SliceReceivedTime"] = slice_received_time;
  ret["SliceSentTime"] = slice_sent_time;
  ret["Time"] = time_us;
  return ret;
}

json generated_algorithm_t::to_json() {
  json ret;
  ret["balance_ratio"] = balance_ratio;
  ret["chunk_size_B"] = chunk_size_B;
  assert(sketch != nullptr);
  ret["sketch"] = sketch->to_json();
  assert(final_schedule != nullptr);
  ret["final_schedule"] = final_schedule->to_json();
  /*
  ret["schedules"] = json::array();
  for (int i = 0; i < schedules.size(); ++i) {
    assert(schedules[i] != nullptr);
    ret["schedules"].push_back(schedules[i]->to_json());
  }
  */
  return ret;
}
