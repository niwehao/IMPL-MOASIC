/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PerfModel/Chunk.h"
#include "PerfModel/Device.h"
#include "PerfModel/Event.h"
#include "PerfModel/Link.h"
#include "PerfModel/Route.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include "common/log.h"
#include <cassert>
#include <memory>

using namespace Simulation;

void Chunk::chunk_arrived_next_device(void* const chunk_ptr) noexcept {
    assert(chunk_ptr != nullptr);

    // cast to unique_ptr<Chunk>
    auto chunk = std::unique_ptr<Chunk>(static_cast<Chunk*>(chunk_ptr));

    // set current node to next_node
    chunk->mark_arrived_next_device();
    auto device = chunk->currentDevice();
    if (device->type == NODE_GPU) {
        assert(chunk->dst_gpu == device->get_local_id());
    }

    if (chunk->isLastSlice()) {
        chunk->currentDevice()->add_received_data(chunk->data);
    }

    if (chunk->arrived_dest()) {
        // chunk will be destroyed automatically
    } else {
        // send this chunk to next dest
        const auto current_node = chunk->currentDevice();
        current_node->schedule_send(std::move(chunk));  // send chunk to next des
    }
}

std::string Chunk::datatoStr() const {
    std::string data_str;
    for (auto &[gpuid, pos] : data) {
        assert(gpuid >= 0);
        assert(pos >= 0);
        data_str += format("%d#%d|", gpuid, pos);
    }
    return data_str;
}

std::string Chunk::toStr(bool verbose) const {
    std::string data_str = datatoStr();
    std::string res = "Chunk #" + std::to_string(id) + ", data: " + data_str + ", parent: " + std::to_string(parent_id) + ", ";
    res += "Slice: " + std::to_string(slice_id) + "/" + std::to_string(nslices) + ", ";
    res += "send_epoch: " + std::to_string(send_epoch) + ", ";
    if (verbose) {
        // res += "Route: " + route->toStr();
        res += "Size: " + std::to_string(chunk_size) + ", ";
        res += "ArrivalTime: " + std::to_string(arrival_time) + ", ";
        res += "CurrentNode: " + current_node->toStr(false) + ", ";
        if (next_node != nullptr) {
            res += "NextNode: " + next_node->toStr(false) + ", ";
        }
    }
    return res;
}

void Chunk::mark_arrived_next_device() noexcept {
    assert(current_node != nullptr);
    assert(next_node != nullptr);
    assert(current_node != next_node);

    // update current node to next node
    current_node = next_node;
}

Chunk::Chunk(const ChunkSize chunk_size, std::shared_ptr<Route> route, std::shared_ptr<RouteNode> current_node, std::set<Solver::SrcChunk> &data, Solver::SrcChunk data_rep, int slice_id, int nslices, int id, EventTime arrival_time, int src_gpu, int dst_gpu) noexcept
    : id(id), data_rep(data_rep), slice_id(slice_id), nslices(nslices), arrival_time(arrival_time), src_gpu(src_gpu), dst_gpu(dst_gpu), chunk_size(chunk_size), route(route), current_node(current_node) {
    assert(chunk_size > 0);
    assert(this->route != nullptr);
    assert(this->current_node != nullptr);
    parent_id = id;
    this->data = data;
}

std::shared_ptr<RouteNode> Chunk::currentRouteNode() const noexcept {
    assert(current_node != nullptr);
    return current_node;
}

std::shared_ptr<RouteNode> Chunk::nextRouteNode() const noexcept {
    assert(current_node != nullptr);
    return next_node;
}

std::shared_ptr<Device> Chunk::currentDevice() const noexcept {
    assert(current_node != nullptr);
    return current_node->device;
}

bool Chunk::arrived_dest() const noexcept {
    return current_node->next_nodes.empty();
}

ChunkSize Chunk::get_size() const noexcept {
    assert(chunk_size > 0);

    // return chunk size
    return chunk_size;
}

void Chunk::scheduleNextRouteNode(std::shared_ptr<RouteNode> nxt, int epoch, send_type_e type) noexcept {
    assert(nxt != nullptr);
    next_node = nxt;
    send_type = type;
    send_epoch = epoch;
}

std::unique_ptr<Chunk> Chunk::copyChunk(int new_id) noexcept {
    auto new_chunk = std::make_unique<Chunk>(chunk_size, route, current_node, data, data_rep, slice_id, nslices, new_id, arrival_time, src_gpu, dst_gpu);
    new_chunk->parent_id = id;
    return new_chunk;
}

double Chunk::remainingLongestPath() {
    auto node = nextRouteNode();
    return node->max_rem_t;
}

bool ChunkPriority::operator()(const std::unique_ptr<Chunk>& lhs, const std::unique_ptr<Chunk>& rhs) const {
    auto lprio = lhs->send_epoch, rprio = rhs->send_epoch;
    // If the epoch is the same from the specified algorithm, we use a heuristic-based ordering algorithm
    if (lprio == rprio) {
        // Send the chunk with the longest remaining path first
        auto lp = lhs->remainingLongestPath(), rp = rhs->remainingLongestPath();
        if (lp == rp) {
            // For the same remaining path, use the arrival time and slice id to break the tie
            auto larr = lhs->arrival_time, rarr = rhs->arrival_time;
            if (larr == rarr) {
                return lhs->slice_id > rhs->slice_id;
            } else {
                if (lhs->id == rhs->id) {
                    if (!((lhs->slice_id < rhs->slice_id) == (larr < rarr))) {
                        LOOOG(SOLVER, INFO, "ERROR: Chunk %s and %s have same id but different slice ids and arrival times", lhs->toStr().c_str(), rhs->toStr().c_str());
                    }
                    assert((lhs->slice_id < rhs->slice_id) == (larr < rarr));
                }
                return larr > rarr;
            }
        } else {
            if (lhs->id == rhs->id) {
                LOOOG(SOLVER, INFO, "ERROR: Chunk %s and %s have same id", lhs->toStr().c_str(), rhs->toStr().c_str());
                LOOOG(SOLVER, INFO, "lp: %lf, rp: %lf", lp, rp);
                LOOOG(SOLVER, INFO, "Current node: %s and %s", lhs->currentRouteNode()->toStr().c_str(), rhs->currentRouteNode()->toStr().c_str());
                LOOOG(SOLVER, INFO, "Next node: %s and %s", lhs->nextRouteNode()->toStr().c_str(), rhs->nextRouteNode()->toStr().c_str());
                LOOOG(SOLVER, INFO, "Next node ptr: %p and %p", lhs->nextRouteNode().get(), rhs->nextRouteNode().get());
                LOOOG(SOLVER, INFO, "lp: %lf, rp: %lf", lhs->nextRouteNode()->max_rem_t, rhs->nextRouteNode()->max_rem_t);
            }
            assert(lhs->id != rhs->id);
            return lp < rp;
        }
    } else {
        assert(lprio != -1 && rprio != -1);
        assert(lhs->id != rhs->id);
        // Small prio at first since prio is epoch
        return lprio > rprio;
    }
}

std::shared_ptr<Device> Chunk::getNextGpu() const noexcept {
    auto next = nextRouteNode();
    assert(next != nullptr);
    while(next->device->type != NODE_GPU) {
        // TODO: assumption: non-GPU node doesn't copy
        assert(next->next_nodes.size() == 1);
        next = next->next_nodes[0].next;
        assert(next != nullptr);
    }
    return next->device;
}

bool Chunk::isLastSlice() const noexcept {
    return slice_id == nslices - 1;
}

bool Chunk::isFirstSlice() const noexcept {
    return slice_id == 0;
}