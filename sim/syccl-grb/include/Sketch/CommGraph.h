// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sketch/Collective.h"
#include "PerfModel/Type.h"
#include "PerfModel/TopologyLayer.h"
#include <memory>
#include <vector>
#include <string>
#include <cassert>

extern int gid;

enum dst_prune_e
{
    dst_prune_undefine = 0,
    dst_prune_balance,
    dst_prune_imbalance
};

struct layer_prune_t {
    int step_split = -1;
    std::vector<int> num_group_per_step;
    std::vector<int> num_source_per_group_per_step;
    dst_prune_e dst_prune = dst_prune_undefine;
};

struct sched_source_t {
    int step;
    int layer;
    int group;
    std::set<int> sources;
    bool equal(sched_source_t& e1) const {
        return step == e1.step && layer == e1.layer && group == e1.group && sources == e1.sources;
    }
};

struct sched_group_t {
    std::vector<std::pair<std::set<int>, std::set<int>>> src_dest_pairs;
};

struct sched_layer_t {
    std::map<int, sched_group_t> comm_groups;   // Per-group communication
    std::map<int, std::set<int>> config_id_to_groups;
    std::map<int, int> group_to_config_id;
};

struct sched_step_t {
    std::map<int, std::map<int, sched_group_t>> layer_group_comm;
};

struct layer_cand_t {
    int layer = -1;
    std::map<int, std::set<int>> group_sources;
    std::map<int, std::set<int>> group_dests;
};
struct sched_layer_size_t {
    int group_id = -1;
    int used_group_num = 0;
    int comm_source_per_group;        // Comm from other layers, used as source
    int non_comm_source_per_group;    // Comm from other layers, not used as source
    int dest_per_group;     // Comm from current layer (i.e., current group)
    sched_layer_size_t(int g, int s1, int s2, int d, int g_id=-1) : group_id(g_id), used_group_num(g), comm_source_per_group(s1), non_comm_source_per_group(s2), dest_per_group(d) {}
};

struct sched_layer_config_t {
    std::vector<sched_layer_size_t> config;
};

struct topo_subset_t {
    // layer -> group -> (src -> (dsts not including lower levels))
    std::map<int, std::map<int, std::map<int, std::set<int>>>> layer_group_src_dest;
    // gpu -> (layer -> group)
    std::map<int, std::map<int, int>> gpu_layer_group;
};

struct sched_t {
    int id;
    int sender;
    int gpu_num; // including sender
    // Step 1 and 2
    std::map<int, sched_layer_config_t> per_layer_size;
    int tmp_acc_dest;
    topo_subset_t topo;
    // Step 3
    std::map<int, sched_source_t> comm_by_gpus; // The comm source of each GPU
    std::map<int, sched_layer_t> comm_by_layer; // Per-step communication
    std::map<int, sched_step_t> comm_by_step;
    std::map<int, layer_prune_t> layer_prunes;
    std::set<int> cur_step_sources;
    std::set<int> nxt_step_sources;
    [[nodiscard]] std::string toStr() const;
};
int total_num_groups(const sched_t& e, int layer);

struct sched_graph_node_t {
    int id;
    std::pair<std::set<int>, std::set<int>> src_dest_pair;
    int step, layer, group;
    std::vector<int> next, deps;
    sched_graph_node_t(int id, std::pair<std::set<int>, std::set<int>> src_dest_pair, int step, int layer, int group) :
            id(id), src_dest_pair(src_dest_pair), step(step), layer(layer), group(group) {}
    nlohmann::json to_json();
};

struct sched_graph_t {
    int ngpus, src_gpu;
    std::vector<sched_graph_node_t> nodes;
    // which node does this gpu get the data at
    std::vector<int> received_node;
    std::vector<int> a2a_parent;
    std::vector<int> a2a_rem_size;
    sched_graph_t(sched_t &sched);
    sched_graph_t(int ngpus, int src_gpu) : ngpus(ngpus), src_gpu(src_gpu) {}
    sched_graph_t(nlohmann::json &sched);
    void extractA2AInfo();
    std::string toStr();
    nlohmann::json to_json();
};

struct ExtractedLayerInfo {
    // gpu -> layer -> (group_id, group_local_id)
    std::map<int, std::map<int, std::pair<int, int>>> group_idx;
    // layer -> group -> size
    std::map<int, std::map<int, int>> group_size;
    // layer -> group -> maps
    std::map<int, std::map<int, std::map<int, int>>> local_to_global, global_to_local;
    std::set<int> used_layers;
    int complete_layer;
    std::map<int, std::map<int, Simulation::layer_relationship_type>> layer_relation;
    ExtractedLayerInfo(Simulation::topo_info_t &topo_info, std::set<int> used_layers);
    std::pair<int, int> getGroupIdx(int gpu, int layer) const;
    std::pair<int, int> getGroupRelIdx(int gpu, int layer, int last_layer) const;
    int getLayerRelSize(int layer, int last_layer) const {
        assert(group_size.at(layer).at(0) % group_size.at(last_layer).at(0) == 0);
        return group_size.at(layer).at(0) / group_size.at(last_layer).at(0);
    }
    int offsetToOtherGroup(int gpu, int other_group, int layer, int x) const;
    int offsetToRelOtherGroup(int gpu, int other_group, int layer, int last_layer, int x) const;
    int get_prev_layer(int layer) const {
        auto it = used_layers.find(layer);
        return it == used_layers.begin() ? -1 : *std::prev(it);
    }
    int add_complete_layer(int ngpus);
};

struct GpuMapping {
    // (node id -> mapped node id)
    std::map<int, int> node_map;
    // (layer_id -> (group_id -> mapped group id))
    std::map<int, std::map<int, int>> layer_map;
    GpuMapping() = default;
    std::string toStr();
};

GpuMapping genSingleMapping(int x, ExtractedLayerInfo layer_info, std::set<int> used_layers);

struct intra_group_sched {
    // chunk_id -> scheds in terms of GPUid
    std::map<int, std::pair<std::set<int>, std::set<int>>> src_dest_pairs;
    std::string toStr();
    nlohmann::json to_json();
};

struct full_sched_graph_node_t {
    int id;
    int step, layer;
    // group id -> and corresponding source and dest pairs
    std::map<int, intra_group_sched> group_scheds;
    std::vector<int> next, deps;
    std::string toStr();
    nlohmann::json to_json();
};

struct full_sched_graph_t {
    int ngpus;
    std::shared_ptr<sched_graph_t> src_graph;
    std::vector<full_sched_graph_node_t> nodes;
    std::map<int, std::vector<int>> a2a_chunk_parent;
    std::map<int, std::vector<int>> a2a_chunk_rem_size;
    int root_src_gpu;
    std::vector<GpuMapping> gpu_maps;
    std::set<int> used_layers;
    std::string toStr();
    nlohmann::json to_json();
};

struct combined_sketch_t {
    std::vector<std::shared_ptr<full_sched_graph_t>> sketches;
    std::vector<double> chunk_size_ratio;
    void addSketch(std::shared_ptr<full_sched_graph_t> sketch, double ratio) {
        sketches.push_back(sketch);
        chunk_size_ratio.push_back(ratio);
    }
    std::string brief() {
        std::string ret = "Sketches: ";
        for (int i = 0; i < sketches.size(); ++i) {
            ret += sketches[i]->src_graph->toStr() + ", Ratio: " + std::to_string(chunk_size_ratio[i]) + "\n";
        }
        return ret;
    }
    nlohmann::json to_json();
};

struct sketch_solve_workload_t {
    std::shared_ptr<full_sched_graph_t> sketch;
    double balance_ratio;
    Simulation::ChunkSize chunk_size_B;
    int split_chunks;
    int id;
    sketch_solve_workload_t(std::shared_ptr<full_sched_graph_t> sketch, double balance_ratio, Simulation::ChunkSize chunk_size_B, int split_chunks, int id) :
            sketch(sketch), balance_ratio(balance_ratio), chunk_size_B(chunk_size_B), split_chunks(split_chunks), id(id) {}
};

struct solved_schedule_t {
    std::shared_ptr<Solver::SimResult> schedule;
    uint64_t chunk_size_B;
    double parse_time_ms, solver_time_ms, sim_time_ms;
    int ndemands, nsolves;
    std::string metaToStr() {
        return "Chunk Size: " + std::to_string(chunk_size_B) + "B, Parse Time: " + std::to_string(parse_time_ms) + "ms, Solver Time: " + std::to_string(solver_time_ms) + "ms, Sim Time: " + std::to_string(sim_time_ms) + "ms, Demands: " + std::to_string(ndemands) + ", Solves: " + std::to_string(nsolves);
    }
    nlohmann::json to_json();
};

struct generated_algorithm_t {
    double balance_ratio;
    uint64_t chunk_size_B;
    std::shared_ptr<combined_sketch_t> sketch;
    std::shared_ptr<Solver::SimResult> final_schedule;
    std::vector<std::shared_ptr<solved_schedule_t>> schedules;
    nlohmann::json to_json();
};

struct demand_map_t {
    std::map<int, int> gpu_map, chunk_map;
    bool checkValid();
    std::string toStr() {
        std::string ret = "GPU Map: {";
        for (auto &[a, b] : gpu_map) {
            ret += std::to_string(a) + " -> " + std::to_string(b) + ", ";
        }
        ret += "}, Chunk Map: {";
        for (auto &[a, b] : chunk_map) {
            ret += std::to_string(a) + " -> " + std::to_string(b) + ", ";
        }
        ret += "}";
        return ret;
    }
};

struct sched_graph_dup_t {
    static int total_dup;
    int ngpus, src_gpu;
    int id, prev_id;
    const std::shared_ptr<sched_graph_t> graph;
    // std::vector<sched_graph_t> dup_graphs;
    std::vector<GpuMapping> gpu_maps;
    std::vector<std::map<int, std::map<int, int>>> layer_map;
    std::queue<int> node_queue;
    std::vector<int> in_deg;
    sched_graph_dup_t(const std::shared_ptr<sched_graph_t> graph, const ExtractedLayerInfo &layer_info);
    // Handwritten copy constructor for id and prev_id, remember to update if more fields are added
    sched_graph_dup_t(const sched_graph_dup_t &other);
    std::string toStr();
};

class CommGraphNode {
public:
    chunk_t chunk;

    std::map<int, CommGraphNode*> comm_next;
    std::map<int, CommGraphNode*> comm_prev;
    std::set<CommGraphNode*> next_nodes;
    std::set<CommGraphNode*> prev_nodes;

    std::map<int, CommGraphNode*> comp_next;
    std::map<int, CommGraphNode*> comp_prev;

    CommGraphNode(chunk_t chunk);

    CommGraphNode* getPrevNode(int gpu_id);
    CommGraphNode* getNextNode(int gpu_id);

    void connectToNext(CommGraphNode* node);

    std::string toStr(bool prev, bool next);
};

static int construct_graph_num;

class CommGraph {
public:
    sched_t ec_mapping;
    int id;

    std::set<int> tmp_remained_gpus; // Used for searching algorithms
    // std::map<int, basic_algo_e> per_layer_algo;

    std::map<int, CommGraphNode*> gpu_nodes;
    std::set<CommGraphNode*> root_nodes;
    std::set<CommGraphNode*> end_nodes;

    std::pair<std::set<CommGraphNode*>, std::vector<int>> getNextLayerNodes(std::set<CommGraphNode*> nodes);

    CommGraph(sched_t& ec_mapping) : ec_mapping(ec_mapping), id(construct_graph_num++) {

    };
    CommGraph(sched_t& ec_mapping, std::set<int>& receiver_gpus, chunk_t sender_chunk) :
            ec_mapping(ec_mapping), id(construct_graph_num++), tmp_remained_gpus(receiver_gpus) {
        tmp_remained_gpus.insert(sender_chunk.gpu_id);
        addGraphNode(sender_chunk);
        construct_graph_num++;
    }
    CommGraph(CommGraph* graph) :
            ec_mapping(graph->ec_mapping), id(construct_graph_num++),
            tmp_remained_gpus(graph->tmp_remained_gpus) /*, per_layer_algo(graph->per_layer_algo)*/ {
        construct_graph_num++;
    };

    // std::map<CommGraphNode*, CommGraphNode*> mergeWithGraph(CommGraph* graph);

    CommGraphNode* addGraphNode(chunk_t chunk, CommGraphNode* comm_prev= nullptr);
    ~CommGraph();

    std::string toStr(bool detail=false);
};

typedef CommGraph* CommGraph_t;
