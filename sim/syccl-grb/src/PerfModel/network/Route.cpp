// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "Sketch/CommGraph.h"
#include "PerfModel/Route.h"
#include "PerfModel/Link.h"
#include "PerfModel/Type.h"
#include "Solver/common.h"
#include <cassert>
#include <memory>
#include <queue>

using namespace Simulation;

void RouteNode::connectNextNode(SendInstruction &next) {
    next_nodes.push_back(next);
}

std::string RouteNode::toStr(bool verbose) {
    std::string res = "Device: " + device->toStr() + ", Chunk: " + format("%d#%d", chunk.first, chunk.second);
    if (!verbose) {
        return res;
    }
    res += "\n";
    if (!next_nodes.empty()) {
        res += "\tNextNodes: ";
        for (auto inst : next_nodes) {
            res += inst.next->device->toStr() + "@e" + std::to_string(inst.prio) + ", ";
        }
        res += "\n";
    }
    return res;
}

std::string Route::toStr() {
    std::string res = "Full Route:\n";
    for (auto root : roots) {
        res += "Root Node: " + root->toStr();
        std::map<std::shared_ptr<RouteNode>, bool> visited;
        std::queue<std::shared_ptr<RouteNode>> queue;
        queue.push(root);
        visited[root] = true;
        int i = 0;
        while (!queue.empty()) {
            auto cur = queue.front();
            queue.pop();
            res += "Node " + std::to_string(i++) + ": " + cur->toStr();
            for (auto inst : cur->next_nodes) {
                auto next = inst.next;
                if (visited.find(next) == visited.end()) {
                    queue.push(next);
                    visited[next] = true;
                }
            }
        }
    }
    return res;
}

std::shared_ptr<Simulation::Link> Simulation::getLink(std::shared_ptr<RouteNode> src, std::shared_ptr<RouteNode> dest) {
    auto src_dev = src->device, dest_dev = dest->device;
    auto dest_id = dest_dev->get_id();
    assert(src_dev->connected(dest_id));
    return src_dev->links[dest_id];
}

void Route::dfsRouteDepth(std::shared_ptr<RouteNode> cur, std::map<std::shared_ptr<RouteNode>, bool> &visited, ChunkSize chunk_size) {
    if (visited.find(cur) != visited.end()) {
        return;
    }
    visited[cur] = true;
    cur->max_rem_t = 0;
    for (auto inst : cur->next_nodes) {
        auto next = inst.next;
        dfsRouteDepth(next, visited, chunk_size);
        auto link = getLink(cur, next);
        // TODO: cosider SWITCH_TO_X links here?
        auto ndep = next->max_rem_t + link->getLatencyns() + chunk_size * link->getInvBandwidthnspB();
        cur->max_rem_t = std::max(cur->max_rem_t, ndep);
    }
}

Route::Route(std::shared_ptr<Topology> topo, std::shared_ptr<Solver::EventSolution> sol, bool PXN): topology(topo) {
    auto &chunk_sizes = sol->chunk_sizes;
    for (auto [src_c, sends] : sol->events) {
        // every GPU in the send tree correspond to one RouteNode
        std::map<int, std::shared_ptr<RouteNode>> node_map;
        auto inits = sol->init_devs[src_c];
        for (auto src_dev : inits) {
            auto route_node = std::make_shared<RouteNode>(src_c, topology->devices[src_dev]);
            assert(topology->devices[src_dev]->type == NODE_GPU);
            node_map[topology->devices[src_dev]->get_local_id()] = route_node;
            roots.push_back(route_node);
        }
        for (auto event : sends) {
            int src = event.src_gpu, dst = event.dst_gpu, epoch = event.epoch;
            auto dst_dev = topology->getNpuDevice(dst), src_dev = topology->getNpuDevice(src);
            if (node_map.find(dst) == node_map.end()) {
                auto dst_node = std::make_shared<RouteNode>(src_c, dst_dev);
                node_map[dst] = dst_node;
            }
            if (node_map.find(src) == node_map.end()) {
                auto src_node = std::make_shared<RouteNode>(src_c, src_dev);
                node_map[src] = src_node;
            }
            auto dst_node = node_map[dst];
            // Find the complete route between two GPUs containing network nodes
            // Note that this routing algorithm only works for certain topologies and needs to be extended for other topologies
            auto route_seg = topology->getNpuRoute(src, dst, event.layer_used, PXN);
            if (route_seg.size() > 2) {
                // if switches are included, then there must be an even number of switches
                assert(route_seg.size() % 2 == 1);
            }
            auto last_node = node_map[src];
            for (int i = 1; i < route_seg.size() - 1; i++) {
                auto interm_node = std::make_shared<RouteNode>(src_c, route_seg[i]);
                assert(last_node->device->connected(interm_node->device->get_id()));
                auto send_type = get_send_type(last_node->device->type, interm_node->device->type);
                auto inst = SendInstruction(epoch, interm_node, send_type);
                last_node->connectNextNode(inst);
                last_node = interm_node;
            }
            auto send_type = get_send_type(last_node->device->type, dst_node->device->type);
            auto inst = SendInstruction(epoch, dst_node, send_type);
            assert(last_node->device->connected(dst_dev->get_id()));
            last_node->connectNextNode(inst);
        }
    }
    // Calculate the maximum remaining time for each node in the route tree, so that we can queue sends based on the bottleneck later
    std::map<std::shared_ptr<RouteNode>, bool> visited;
    for (auto root : roots) {
        assert(chunk_sizes.count(root->chunk));
        auto chunk_size = chunk_sizes[root->chunk];
        dfsRouteDepth(root, visited, chunk_size);
    }
}