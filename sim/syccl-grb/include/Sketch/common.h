// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ratio>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <queue>


struct algo_solve_config_t {
    double filter_balance_ratio = 4;
    double solve_balance_ratio = 0.2;
    bool do_raw_solve = true;
    double raw_solve_balance_ratio = 0.2;
    // always fine-grain solve the top k algorithms
    int num_used_algos_min = 4, num_used_algos_max = 8;
    // fine-grain solve all algorithms that are within time_diff_threshold of the best algorithm
    double time_diff_threshold = 0.2;
    std::string solve_output = "solve_result.json";
};

struct prune_policy_t {
    /* Step 1: Layer combinations */
    int max_comb_layer_num = -1;        // maximum num_layers allowed in an algorithm
    std::vector<int> ignored_layers;
    std::vector<int> must_contain_layers;
    /* Step 2: Number of source and destination per group */
    std::map<int, std::set<int>> layer_comm_ordering; // L1:[L2] means L1 should have more communication than L2
    std::map<int, int> layer_comm_uplimit;            // L:N means at most N GPUs can receive chunk from layer L
    bool all_groups_used;                             // All layers use all groups
    // sources to explore at each step
    // const bool prefer_equal_source_per_group = true;  // Groups in the same layer have same number of sources
    bool allow_unequal_source_per_group = true;       // Groups in the same layer have same number of sources (except sources that do not communicate)
    // destinations to explore at each step
    // const bool prefer_equal_dest_per_source = true;       // Each source has same number of destinations
    bool allow_unequal_dest_per_source = true;            // Each source has same number of destinations (except sources that do not communicate)
    bool at_least_one_layer_no_source_with_0_dest = true; // At least one layer has all sources to communicate
    bool allow_source_group_special = false;               // Allow source Groups has different behaviors from other groups
    /* Step 3: steps, sources and destnations in each group */
    int limit_num_steps = -1;                         // Uplimit of num_steps, ignored if =-1
    float limit_num_steps_ratio = -1;                 // Uplimit of num_steps = ratio * total num_groups, ignored if =-1 or limit_num_steps!=-1
    // determine possible sources
    // const bool only_next_step_source = true;        // A dest GPU from step S can only be used as source at step S+1
    bool start_source_diff_layer = true;            // A destination GPU from (layer L, step S) should not be used as source in (layer L, step S+1)
    bool same_step_diff = true;                     // A layer should have communication at step x, x+N, x+2N, ...
    // source combinations
    //const bool partition_source_groups = true;      // Partition sources into equivalent groups
    // per-layer num_sources and num_groups cross
    bool prune_sources_by_compare_steps = true;
    bool allow_num_groups_arithmetic_per_step = true;       // num groups cross step should be X, X*N, X*N*N, ...
    bool allow_source_per_group_arithmetic_per_step = true; // num groups cross step should be X, X+N, X+2N, ...
    bool allow_num_groups_geometric_per_step = true;
    bool allow_source_per_group_geometric_per_step = true;
    bool drop_small_groups = true;                // at each step, try to use more groups
    bool drop_small_sources = true;               // at each step, try to use more source_per_group
    bool keep_large_groups = false;
    bool keep_large_sources = false;
    // determine specific sources and destinations per group
    bool partition_sources = true;
    bool partition_dests = true;
    // determined destinations
    bool prune_dests = true;
    bool allow_dest_balanced = true;
    bool allow_dest_centralized = true;
    int max_dest_combs = 100;
};

class Clock_ {
public:
    double tic() {
        auto end_time = std::chrono::high_resolution_clock::now();
        double ret_ms = std::chrono::duration<double, std::nano>(end_time - start_time).count() / 1e6;
        start_time = std::chrono::high_resolution_clock::now();
        return ret_ms;
    }

    Clock_() {
        start_time = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

class BipartiteGraph {
    const int NIL = 0;
    const int INF = std::numeric_limits<int>::max();
    int m, n; // number of vertices on each side
    std::vector<int> *adj; // adjacency list
    std::vector<int> pair_U, pair_V, dist;
    
public:
    BipartiteGraph(int m, int n) : m(m), n(n) {
        adj = new std::vector<int>[m+1];
        pair_U = std::vector<int>(m+1, NIL);
        pair_V = std::vector<int>(n+1, NIL);
        dist = std::vector<int>(m+1);
    }
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }
    
    bool bfs() {
        std::queue<int> Q;
        
        for (int u = 1; u <= m; u++) {
            if (pair_U[u] == NIL) {
                dist[u] = 0;
                Q.push(u);
            } else {
                dist[u] = INF;
            }
        }
        
        dist[NIL] = INF;
        
        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            
            if (dist[u] < dist[NIL]) {
                for (int v : adj[u]) {
                    if (dist[pair_V[v]] == INF) {
                        dist[pair_V[v]] = dist[u] + 1;
                        Q.push(pair_V[v]);
                    }
                }
            }
        }
        
        return (dist[NIL] != INF);
    }
    
    bool dfs(int u) {
        if (u != NIL) {
            for (int v : adj[u]) {
                if (dist[pair_V[v]] == dist[u] + 1) {
                    if (dfs(pair_V[v])) {
                        pair_V[v] = u;
                        pair_U[u] = v;
                        return true;
                    }
                }
            }
            dist[u] = INF;
            return false;
        }
        return true;
    }
    
    int hopcroftKarp() {
        int matching = 0;
        
        while (bfs()) {
            for (int u = 1; u <= m; u++) {
                if (pair_U[u] == NIL && dfs(u)) {
                    matching++;
                }
            }
        }
        
        return matching;
    }

    std::vector<int> getMatchesFromU() {
        return pair_U;
    }
};
