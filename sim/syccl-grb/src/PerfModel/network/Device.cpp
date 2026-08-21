/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PerfModel/Device.h"
#include "PerfModel/Chunk.h"
#include "PerfModel/Link.h"
#include "PerfModel/Route.h"
#include "Solver/common.h"
#include "Sketch/common.h"
#include <cassert>

using namespace Simulation;

Device::Device(DeviceId id, DeviceId local_id) noexcept : device_id(id), local_id(local_id) {
    assert(device_id >= 0);
    assert(local_id >= 0);
}

std::string Device::toStr() const {
    std::string type_str;
    switch (type) {
        case NODE_GPU:
            type_str = "GPU";
            break;
        case NODE_NIC:
            type_str = "NIC";
            break;
        case NODE_SWITCH_NV:
            type_str = "NVSw";
            break;
        case NODE_SWITCH_NET:
            type_str = "NetSw";
            break;
        default:
            type_str = "Unknown";
            break;
    }
    return type_str + "/" + std::to_string(local_id);
}

DeviceId Device::get_id() const noexcept {
    assert(device_id >= 0);
    return device_id;
}

// Send according to all SendInstructions in the current RouteNode
void Device::schedule_send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    assert(chunk->currentDevice()->get_id() == device_id);
    if (chunk->arrived_dest()) {
        return;
    }

    // copy and send
    const auto node = chunk->currentRouteNode();
    if (this->type == NODE_GPU) {
        chunk->src_gpu = local_id;
    } else {
        assert(chunk->src_gpu != -1 && chunk->dst_gpu != -1);
    }
    for (const auto &inst : node->next_nodes) {
        const int prio = inst.prio;
        const auto next_node = inst.next;
        const send_type_e type = inst.type;
        const auto next_dest = next_node->device;
        const auto next_dest_id = next_dest->get_id();
        assert(connected(next_dest_id));
        // send the chunk to the next dest, creating a new chunk for every send as the chunk is copied to multiple destinations
        // For pipelined send, all slices should have the same id so we have to record the id here
        if (!outstanding_chunk_id_map.count({chunk->id, next_node})) {
            assert(chunk->isFirstSlice());
            outstanding_chunk_id_map[{chunk->id, next_node}] = *num_chunks;
            (*num_chunks)++;
        }
        int new_chunk_id = outstanding_chunk_id_map[{chunk->id, next_node}];
        if (chunk->isLastSlice()) {
            outstanding_chunk_id_map.erase({chunk->id, next_node});
        }
        // delegate this task to the link
        auto new_chunk = chunk->copyChunk(new_chunk_id);
        // update chunk metadata
        new_chunk->scheduleNextRouteNode(next_node, prio, type);
        if (this->type == NODE_GPU) {
            new_chunk->dst_gpu = new_chunk->getNextGpu()->get_local_id();
        }
        auto link = links[next_dest_id];
        bool link_free = !link->pending_chunk_exists();
        link->add_pending_chunk(std::move(new_chunk));
        // To prevent multiple scheduled event for a same batch of chunks
        if (link_free) {
            // This event has a lower priority than chunk arrival, Thus link starts to schedule transmissions only after all chunks are queued up
            link->schedule_process_pending_transmission();
        }
    }
    // chunk is unique_ptr, will be destroyed automatically
}

std::shared_ptr<Link> Device::connect(const DeviceId id, const Bandwidth bandwidth, const Latency latency) noexcept {
    assert(id >= 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // assert there's no existing connection
    assert(!connected(id));

    // create link
    links[id] = std::make_shared<Link>(bandwidth, latency, device_id, id);
    return links[id];
}

bool Device::connected(const DeviceId dest) const noexcept {
    assert(dest >= 0);

    // check whether the connection exists
    return links.find(dest) != links.end();
}

void Device::clear_received_data() noexcept {
    received_data.clear();
}

void Device::clear_chunk_id_map() noexcept {
    outstanding_chunk_id_map.clear();
}

void Device::add_received_data(const std::set<Solver::SrcChunk> &data) noexcept {
    if (type != NODE_GPU) {
        return;
    }
    received_data.push_back(data);
}