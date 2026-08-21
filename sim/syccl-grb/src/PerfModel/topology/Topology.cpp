/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PerfModel/Topology.h"
#include "PerfModel/Link.h"
#include "PerfModel/Type.h"
#include "common/log.h"
#include <cassert>
#include <memory>

using namespace Simulation;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);
    for (auto dev : devices) {
        for (auto &[_, link] : dev->links) {
            link->set_event_queue(event_queue);
        }
    }
}

void Topology::set_simres(std::shared_ptr<Solver::SimResult> res) noexcept {
    assert(res != nullptr);
    for (auto dev : devices) {
        for (auto &[_, link] : dev->links) {
            link->set_simres(res);
        }
    }
}

void Topology::set_num_chunks(int *num_chunks) noexcept {
    assert(num_chunks != nullptr);
    for (auto dev : devices) {
        dev->num_chunks = num_chunks;
    }
}

Topology::Topology() noexcept {
}

std::string NPUInfo::toStr() {
    std::string res = "NPUInfo:\n";
    res += "Device: " + device->toStr() + "\n";
    res += "Machine: " + std::to_string(machine_id) + "\n";
    res += "NIC: " + (NIC == nullptr ? "null" : NIC->toStr()) + "\n";
    res += "NVswitch: " + (NVswitch == nullptr ? "null" : NVswitch->toStr()) + "\n";
    for (int layer = 0; layer < layer_switches.size(); layer++) {
        res += "Netsw-L" + std::to_string(layer) + ": " + layer_switches[layer]->toStr() + (layer == layer_switches.size() - 1 ? "" : "\n");
    }
    return res;
}

void Topology::print() {
    LOOOG(PERFMODEL, TRACE, "================= Full Topology ===================");
    for (auto npu : npus) {
        LOOOG(PERFMODEL, TRACE, npu.toStr().c_str());
    }
}

void Topology::addDevice(node_type_e type) noexcept {
    int id = devices.size(), local_id = dev_count[type]++;
    devices.push_back(std::make_shared<Device>(id, local_id));
    devices.back()->type = type;
    if (type == NODE_GPU) {
        npus.emplace_back(devices.back());
    }
    LOOOG(PERFMODEL, TRACE, "Adding device %s with id %d", devices.back()->toStr().c_str(), id);
}

int Topology::get_devices_count() const noexcept {
    return devices.size();
}

int Topology::get_npus_count() const noexcept {
    return npus.size();
}

void Topology::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    const auto src = chunk->currentDevice()->get_id();
    assert(0 <= src && src < get_devices_count());
    // initiate transmission from src
    devices[src]->schedule_send(std::move(chunk));
}

std::shared_ptr<Device> Topology::getNpuDevice(const int npu_id) noexcept {
    assert(0 <= npu_id && npu_id < get_npus_count());
    assert(npus[npu_id].device != nullptr);
    return npus[npu_id].device;
}

void Topology::connect(const DeviceId src, const DeviceId dest, const Bandwidth bandwidth, const Latency latency,
                       std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>>& links,
                       const bool bidirectional) noexcept {
    assert(0 <= src && src < get_devices_count());
    assert(0 <= dest && dest < get_devices_count());
    LOOOG(PERFMODEL, TRACE, "Connecting %s and %s with bandwidth %f GB/s and latency %f ns", devices[src]->toStr().c_str(), devices[dest]->toStr().c_str(), bandwidth, latency);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // connect src -> dest
    links[devices[src]][devices[dest]] = devices[src]->connect(dest, bandwidth, latency);

    // if bidirectional, connect dest -> src
    if (bidirectional) {
        links[devices[dest]][devices[src]] = devices[dest]->connect(src, bandwidth, latency);
    }
}

std::vector<std::shared_ptr<Device>> Topology::getNpuRoute(int src_npu_id, int dst_npu_id, int layer, bool PXN) noexcept {
    assert(0 <= src_npu_id && src_npu_id < get_npus_count());
    assert(0 <= dst_npu_id && dst_npu_id < get_npus_count());

    auto src_info = npus[src_npu_id], dst_info = npus[dst_npu_id];

    std::vector<std::shared_ptr<Device>> route_src = {src_info.device};
    std::vector<std::shared_ptr<Device>> route_dst = {dst_info.device};

    // TODO: We are hardcoding and assuming layer 1 is nvlink/nvswitch here.
    if (src_info.machine_id == dst_info.machine_id && layer < 2) {
        if (src_npu_id != dst_npu_id) {
            if (src_info.NVswitch != nullptr) {
                route_src.push_back(src_info.NVswitch);
            } else {
                // NVLink
                assert(src_info.device->connected(dst_info.device->get_id()));
            }
        }
    } else {
        if (PXN) {
            int machine_size = topo_info.layer_info[1].groups[0].connected_gpus.size();
            assert(machine_size == 8);
            if (src_npu_id / machine_size != dst_npu_id / machine_size &&
                src_npu_id % machine_size != dst_npu_id % machine_size) {
                int interm_npu_id = (src_npu_id / machine_size) * machine_size + dst_npu_id % machine_size;
                assert(interm_npu_id / machine_size == src_npu_id / machine_size);
                auto interm_route = getNpuRoute(src_npu_id, interm_npu_id, 1, false);
                route_src.insert(route_src.end(), interm_route.begin() + 1, interm_route.end());
                src_npu_id = interm_npu_id;
            }
            assert(dst_npu_id % machine_size == src_npu_id % machine_size);
            src_info = npus[src_npu_id], dst_info = npus[dst_npu_id];
        }
        // we assume this is inter-node and requires NIC->sw path
        assert(src_info.NIC != nullptr && dst_info.NIC != nullptr);
        route_src.push_back(src_info.NIC);
        route_dst.push_back(dst_info.NIC);
        int sw_layers = src_info.layer_switches.size();
        for (int i = 0; i < sw_layers; i++) {
            if (src_info.layer_switches[i]->get_id() == dst_info.layer_switches[i]->get_id()) {
                // same switch layer
                route_src.push_back(src_info.layer_switches[i]);
                break;
            } else {
                // different switch layer
                assert(i + 1 < sw_layers);
                route_src.push_back(src_info.layer_switches[i]);
                route_dst.push_back(dst_info.layer_switches[i]);
            }
        }
    }

    route_src.insert(route_src.end(), route_dst.rbegin(), route_dst.rend());
    return route_src;
}

void Topology::resetDevices() {
    for (auto &device : devices) {
        device->clear_received_data();
        device->clear_chunk_id_map();
    }
}
