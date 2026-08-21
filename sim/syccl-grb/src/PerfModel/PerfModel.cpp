// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "PerfModel/PerfModel.h"
#include "PerfModel/Route.h"
#include "Solver/common.h"
#include "Sketch/parse.h"
#include <cassert>
#include <memory>
#include <queue>
#include <regex>
#include <string>

using namespace Simulation;

std::pair<Bandwidth, Latency> parseSpec(std::pair<double, double>& bw_lat_pair) {
    // convert bandwidth from MB/s to GB/s and latency from us to ns
    return {bw_lat_pair.first * 1000.0 * 1000.0 / 1024.0, bw_lat_pair.second * 1000.0};
}

void gen_layers(topo_info_t& topo_info, int layer_id,
                layer_group_type type, std::shared_ptr<Device> switch_device,
                std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>>& links) {
    network_group_t group;
    while (topo_info.layer_info.size() <= layer_id) {
        topo_info.layer_info.emplace_back();
    }
    int group_id = topo_info.layer_info[layer_id].groups.size();
    group.layer_id = layer_id;
    group.group_id = group_id;
    group.type = type;
    group.switch_device = switch_device;
    for (const auto& it : links) {
        auto src_device = it.first;
        if (src_device != switch_device)
            group.devices.insert(src_device);
        topo_info.device_groups[src_device][layer_id] = group_id;
        for (const auto& it1 : it.second) {
            auto dst_device = it1.first;
            auto link = it1.second;
            if (dst_device != switch_device)
                group.devices.insert(dst_device);
            group.links[src_device][dst_device] = link;
        }
    }
    topo_info.layer_info[layer_id].groups.push_back(group);
    if (switch_device != nullptr) topo_info.switch_device_groups[switch_device] = {layer_id, group_id};
}

void gen_singleton_gpu_layer(topo_info_t& topo_info, int layer_id,
                             std::shared_ptr<Device> gpu_device) {
    while (topo_info.layer_info.size() <= layer_id) {
        topo_info.layer_info.emplace_back();
    }
    network_group_t group;
    int group_id = topo_info.layer_info[layer_id].groups.size();
    group.layer_id = layer_id;
    group.group_id = group_id;
    group.type = group_direct;
    group.switch_device = nullptr;
    group.devices.insert(gpu_device);
    topo_info.device_groups[gpu_device][layer_id] = group_id;
    topo_info.layer_info[layer_id].groups.push_back(group);
}

void construct_machine(int gpu_per_machine, int nic_per_machine, int& num_gpus, int& num_nodes, std::string host_type,
                       std::map<DeviceId, int>& devtoGPU,
                       std::vector<node_type_e>& device_type,
                       std::vector<std::map<node_type_e, std::vector<DeviceId>>>& machine_nodes,
                       std::shared_ptr<Topology> topo) {
    std::map<node_type_e, std::vector<DeviceId>> nodes;
    for (int j = 0; j < gpu_per_machine; j++) {
        devtoGPU[num_nodes] = num_gpus;
        nodes[NODE_GPU].push_back(num_nodes);
        device_type.push_back(NODE_GPU);
        topo->addDevice(NODE_GPU);
        num_nodes++, num_gpus++;
    }
    for (int j = 0; j < nic_per_machine; j++) {
        nodes[NODE_NIC].push_back(num_nodes++);
        device_type.push_back(NODE_NIC);
        topo->addDevice(NODE_NIC);
    }
    if (host_type == "nvswitch") {
        nodes[NODE_SWITCH_NV].push_back(num_nodes++);
        device_type.push_back(NODE_SWITCH_NV);
        topo->addDevice(NODE_SWITCH_NV);
    }
    machine_nodes.push_back(nodes);
}

void construct_switch_layer(int layer_id, int switch_num, int& num_nodes,
                            std::vector<node_type_e>& device_type,
                            std::vector<std::vector<DeviceId>>& per_layer_switches,
                            std::shared_ptr<Topology> topo) {
    per_layer_switches.push_back({});
    for (int i = 0; i < switch_num; i++) {
        per_layer_switches[layer_id].push_back(num_nodes++);
        device_type.push_back(NODE_SWITCH_NET);
        topo->addDevice(NODE_SWITCH_NET);
    }
}

void connect_gpus_by_nvswitch(int machine_num, int layer_id, std::pair<double, double>& link_spec,
                              std::map<DeviceId, int>& devtoGPU,
                              std::vector<std::map<node_type_e, std::vector<DeviceId>>>& machine_nodes,
                              std::shared_ptr<Topology> topo) {
    for (int group_id = 0; group_id < machine_num; group_id++) {
        LOOOG(PERFMODEL, TRACE, "Connecting GPUs in machine %d", group_id);
        std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
        if (machine_nodes[group_id][NODE_GPU].size() == 1) {
            int gpu = machine_nodes[group_id][NODE_GPU][0];
            int id = devtoGPU[gpu];
            topo->npus[id].machine_id = group_id;
            gen_singleton_gpu_layer(topo->topo_info, layer_id,
                                    topo->devices[gpu]);
            continue;
        }
        for (auto gpu : machine_nodes[group_id][NODE_GPU]) {
            auto sw = *machine_nodes[group_id][NODE_SWITCH_NV].begin();
            auto [bw, lat] = parseSpec(link_spec);
            topo->connect(gpu, sw, bw, lat, layer_links, true);
            int id = devtoGPU[gpu];
            topo->npus[id].NVswitch = topo->devices[sw];
            topo->npus[id].machine_id = group_id;
        }
        gen_layers(topo->topo_info, layer_id, group_switch, topo->devices[*machine_nodes[group_id][NODE_SWITCH_NV].begin()], layer_links);
    }
}

void connect_gpus_by_nvlink(int machine_num, int layer_id, std::vector<std::vector<DeviceId>>& matrix,
                            std::pair<double, double>& link_spec,
                            std::map<DeviceId, int>& devtoGPU,
                            std::vector<std::map<node_type_e, std::vector<DeviceId>>>& machine_nodes,
                            std::shared_ptr<Topology> topo) {
    for (int group_id = 0; group_id < machine_num; group_id++) {
        LOOOG(PERFMODEL, TRACE, "Connecting GPUs in machine %d", group_id);
        std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
        if (machine_nodes[group_id][NODE_GPU].size() == 1) {
            int gpu = machine_nodes[group_id][NODE_GPU][0];
            int id = devtoGPU[gpu];
            topo->npus[id].machine_id = group_id;
            gen_singleton_gpu_layer(topo->topo_info, layer_id,
                                    topo->devices[gpu]);
            continue;
        }
        // matrix rows are local GPU indices within one machine.
        assert(matrix.size() == machine_nodes[group_id][NODE_GPU].size());
        for (int s = 0; s < matrix.size(); s++) {
            int sender = machine_nodes[group_id][NODE_GPU][s];
            for (int r : matrix[s]) {
                assert(r >= 0 && (size_t)r < machine_nodes[group_id][NODE_GPU].size());
                int receiver = machine_nodes[group_id][NODE_GPU][r];
                auto [bw, lat] = parseSpec(link_spec);
                topo->connect(sender, receiver, bw, lat, layer_links, false);
            }
            // BUGFIX 2026-07-08: npus is indexed by GPU ordinal, not by the
            // global DeviceId; writing npus[sender] corrupted the heap for
            // any machine whose GPU DeviceIds exceed the GPU count (the
            // nvswitch branch above already translates via devtoGPU).
            topo->npus[devtoGPU[sender]].machine_id = group_id;
        }
        gen_layers(topo->topo_info, layer_id, group_direct, nullptr, layer_links);
    }
}

void connect_local_gpus(int layer_id, std::pair<double, double>& link_spec, std::shared_ptr<Topology> topo) {
    for (auto gpu : topo->npus) {
        std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
        auto id = gpu.device->get_id();
        auto [bw, lat] = parseSpec(link_spec);
        topo->connect(id, id, bw, lat, layer_links, false);
        gen_layers(topo->topo_info, layer_id, group_local, nullptr, layer_links);
    }
}


void connecte_gpu_to_nics(int nic_per_machine, int gpu_per_machine, int machine_num, int layer_id, std::pair<double, double>& link_spec,
                          std::vector<std::map<node_type_e, std::vector<DeviceId>>>& machine_nodes,
                          std::map<DeviceId, int>& devtoGPU, std::map<DeviceId, std::vector<int>>& NICtoGPUs,
                          std::shared_ptr<Topology> topo) {
    assert(nic_per_machine > 0);
    int gpu_per_nic = gpu_per_machine / nic_per_machine;
    for (int nic_id = 0; nic_id < nic_per_machine; nic_id++) {
        LOOOG(PERFMODEL, TRACE, "Connecting GPUs to NIC %d", nic_id);
        for (int machine_id = 0; machine_id < machine_num; machine_id++) {
            std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
            auto nic = machine_nodes[machine_id][NODE_NIC][nic_id];
            auto gpu_start = machine_nodes[machine_id][NODE_GPU].begin() + nic_id * gpu_per_nic;
            std::vector<DeviceId> gpus(gpu_start, gpu_start + gpu_per_nic);
            for (auto gpu : gpus) {
                auto [bw, lat] = parseSpec(link_spec);
                topo->connect(gpu, nic, bw, lat, layer_links, true);
                int id = devtoGPU[gpu];
                topo->npus[id].NIC = topo->devices[nic];
                NICtoGPUs[nic].push_back(id);
            }
            gen_layers(topo->topo_info, layer_id, group_switch, topo->devices[nic], layer_links);
        }
    }
}

void connect_nic_to_switches(int nic_per_machine, int machine_num, int layer_id,
                             std::string switch_type, std::pair<double, double>& link_spec,
                             std::vector<std::map<node_type_e, std::vector<DeviceId>>>& machine_nodes,
                             std::map<DeviceId, std::vector<int>>& NICtoGPUs,
                             std::vector<std::map<DeviceId, std::vector<int>>>& NetswtoGPUs,
                             std::vector<std::vector<DeviceId>>& per_layer_switches,
                             std::shared_ptr<Topology> topo) {
    int total_nic_per_switch = nic_per_machine*machine_num / per_layer_switches[layer_id].size();
    int per_switch_nic_per_switch = total_nic_per_switch / machine_num;
    if (switch_type == "multirail") {
        assert(total_nic_per_switch % machine_num == 0);
        for (int switch_id = 0; switch_id < per_layer_switches[layer_id].size(); switch_id++) {
            std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
            LOOOG(PERFMODEL, TRACE, "Connecting NICs to switch %d", switch_id);
            auto sw = per_layer_switches[layer_id][switch_id];
            std::vector<DeviceId> nics;
            for (int machine_id = 0; machine_id < machine_num; machine_id++)
                std::copy(machine_nodes[machine_id][NODE_NIC].begin() + switch_id*per_switch_nic_per_switch,
                          machine_nodes[machine_id][NODE_NIC].begin() + (switch_id+1)*per_switch_nic_per_switch, std::back_inserter(nics));

            for (auto nic : nics) {
                auto [bw, lat] = parseSpec(link_spec);
                topo->connect(nic, sw, bw, lat, layer_links, true);
                for (auto gpu : NICtoGPUs[nic]) {
                    topo->npus[gpu].layer_switches.push_back(topo->devices[sw]);
                    NetswtoGPUs[layer_id][sw].push_back(gpu);
                }
            }
            gen_layers(topo->topo_info, layer_id, group_switch, topo->devices[sw], layer_links);
        }
    } else if (switch_type == "pod") {
        int machine_per_switch = total_nic_per_switch / nic_per_machine;
        assert(total_nic_per_switch % nic_per_machine == 0);
        int machine_id = 0;
        for (int switch_id = 0; switch_id < per_layer_switches[layer_id].size(); switch_id++) {
            std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
            LOOOG(PERFMODEL, TRACE, "Connecting NICs to switch %d", switch_id);
            auto sw = per_layer_switches[layer_id][switch_id];
            std::vector<DeviceId> nics;
            for (int i = 0; i < machine_per_switch; ++i) {
                std::copy(machine_nodes[machine_id][NODE_NIC].begin(),
                          machine_nodes[machine_id][NODE_NIC].end(), std::back_inserter(nics));
                machine_id++;
            }

            for (auto nic : nics) {
                auto [bw, lat] = parseSpec(link_spec);
                topo->connect(nic, sw, bw, lat, layer_links, true);
                for (auto gpu : NICtoGPUs[nic]) {
                    topo->npus[gpu].layer_switches.push_back(topo->devices[sw]);
                    NetswtoGPUs[layer_id][sw].push_back(gpu);
                }
            }
            gen_layers(topo->topo_info, layer_id, group_switch, topo->devices[sw], layer_links);
        }
    } else assert(0);

}

void connect_switches(int layer_id, std::string switch_type, std::pair<double, double>& link_spec,
                      std::vector<std::map<DeviceId, std::vector<int>>>& NetswtoGPUs,
                      std::vector<std::vector<DeviceId>>& per_layer_switches,
                      std::shared_ptr<Topology> topo) {
    if (switch_type == "pod") {
        int switch_per_switch = per_layer_switches[layer_id-1].size() / per_layer_switches[layer_id].size();
        for (int switch_id = 0; switch_id < per_layer_switches[layer_id].size(); switch_id++) {
            std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> layer_links;
            std::vector<DeviceId> low_layer_switch(per_layer_switches[layer_id-1].begin() + switch_id * switch_per_switch,
                                                   per_layer_switches[layer_id-1].begin() + (switch_id + 1) * switch_per_switch);
            auto high_sw = per_layer_switches[layer_id][switch_id];
            for (auto sw : low_layer_switch) {
                auto [bw, lat] = parseSpec(link_spec);
                topo->connect(sw, high_sw, bw, lat, layer_links, true);
                for (auto gpu : NetswtoGPUs[layer_id-1][sw]) {
                    topo->npus[gpu].layer_switches.push_back(topo->devices[high_sw]);
                    NetswtoGPUs[layer_id][high_sw].push_back(gpu);
                }
            }
            gen_layers(topo->topo_info, layer_id, group_switch, topo->devices[high_sw], layer_links);
        }
    } else {
        assert(false);
    }
}

std::pair<Bandwidth, Latency> parseSpec(double bw_mbpus, double lat_us) {
    // convert bandwidth from MB/s to GB/s and latency from us to ns
    return {bw_mbpus * 1000.0 * 1000.0 / 1024.0, lat_us * 1000.0};
}

std::shared_ptr<Topology> Simulation::constructTopology(nlohmann::json& config) {
    int num_gpus = 0;
    int num_nodes = 0;
    std::vector<std::map<node_type_e, std::vector<DeviceId>>> machine_nodes;
    std::vector<node_type_e> device_type;
    std::map<DeviceId, int> devtoGPU;
    std::map<DeviceId, std::vector<int>> NICtoGPUs;
    std::vector<std::map<DeviceId, std::vector<int>>> NetswtoGPUs;
    std::vector<std::vector<DeviceId>> per_layer_switches;

    std::shared_ptr<Topology> topo = std::make_shared<Topology>();

    LOOOG(PERFMODEL, TRACE, "Constructing topology...");

    int machine_num, gpu_per_machine, nic_per_machine;
    std::string host_type;
    parse_host_config(config, machine_num, gpu_per_machine, nic_per_machine, host_type);
    for (int i = 0; i < machine_num; i++) {
        LOOOG(PERFMODEL, TRACE, "Constructing machine %d", i);
        // Add Nodes inside each machine
        construct_machine(gpu_per_machine, nic_per_machine, num_gpus, num_nodes, host_type,devtoGPU, device_type, machine_nodes, topo);
    }

    int layer_num = parse_topo_layer_num(config);
    NetswtoGPUs.resize(layer_num);
    per_layer_switches.resize(layer_num);

    std::string prev_layer_type = "";
    for (int layer_id = 0; layer_id < layer_num; layer_id++) {
        std::string layer_type = parse_topo_layer_type(config, layer_id);
        std::pair<double, double> link_spec = parse_link_spec(config, layer_id);
        if (layer_type == "local") {
            assert(layer_id == 0);
            connect_local_gpus(layer_id, link_spec, topo);
        } else if (layer_type == "host") {
            if (host_type == "nvswitch") {
                connect_gpus_by_nvswitch(machine_num, layer_id, link_spec, devtoGPU, machine_nodes, topo);
            } else if (host_type == "nvlink") {
                std::vector<std::vector<DeviceId>> matrix = parse_nvlink_matrix(config);
                connect_gpus_by_nvlink(machine_num, layer_id, matrix, link_spec, devtoGPU, machine_nodes, topo);
            } else assert(0); // TODO: connect Mesh/Torus here
        } else if (layer_type == "nic") {
            assert(prev_layer_type == "host");
            connecte_gpu_to_nics(nic_per_machine, gpu_per_machine, machine_num, layer_id, link_spec, machine_nodes, devtoGPU, NICtoGPUs, topo);
        } else if (layer_type == "switch") {
            assert(prev_layer_type == "nic" or prev_layer_type == "switch");
            int switch_num;
            std::string switch_topo;
            parse_switch_config(config, layer_id, switch_num, switch_topo, link_spec);
            LOOOG(PERFMODEL, TRACE, "Constructing switch layer with %d switches", switch_num);
            construct_switch_layer(layer_id, switch_num, num_nodes, device_type, per_layer_switches, topo);
            LOOOG(PERFMODEL, TRACE, "Connecting switches in layer %d", layer_id);
            if (prev_layer_type == "nic") {
                // Multirail / Pod
                connect_nic_to_switches(nic_per_machine, machine_num, layer_id, switch_topo, link_spec, machine_nodes, NICtoGPUs, NetswtoGPUs, per_layer_switches, topo);
            } else {
                connect_switches(layer_id, switch_topo, link_spec, NetswtoGPUs, per_layer_switches, topo);
            }
        } else assert(0); // TODO: Mesh Layer between hosts?
        prev_layer_type = layer_type;
    }

    assert(topo->get_npus_count() == num_gpus);
    assert(topo->get_devices_count() == num_nodes);
    topo->print();
    return topo;
}

void PerfModel::initAnalysis() {
    topo->generateLayerInfo();
}

void PerfModel::resetAnalysis() {
    // Clear received data
    topo->resetDevices();
    num_chunks = 0;
}

#define USE_PIPELINED_MODEL
std::shared_ptr<Solver::SimResult> PerfModel::runAnalysis(const SimConfig &config) {
    LOOOG(PERFMODEL, TRACE, "Running analysis #%d", num_runs++);
    resetAnalysis();
    LOOOG(PERFMODEL, TRACE, "config: %s", config.sol->toStr().c_str());
    const auto event_queue = std::make_shared<EventQueue>();
    topo->set_event_queue(event_queue);
    topo->set_num_chunks(&num_chunks);
    assert(config.sol != nullptr);
/*
    Here we translate the GPU sends without routing info (As in MSCCL algorithm desctription) into full routes
    In theory, the synthesized solution contains the intermediate routing information and should be included in the collective algorithm description
    But this is dependent on the network infrastructure(e.g. how much control do we have over network routes in the CCL) and requires dedicated abstractions included in the CCL
    So, assuming the MSCCL primitive abstraction, we only consider an algorithm as GPU->GPU sends here
    And we model the clos topology as one-big-switch with the aggregated bandwidth(as an ideal model) to exclude the impact of network routing to the algorithm performance
    However, Extending the sketch and the solver to consider routing decisions in switches is always possible
    PXN is considered here as the GPU is used for routing only in PXN and may receive the data twice, which may mess up the Route construction
*/
    std::shared_ptr<Route> route = std::make_shared<Route>(topo, config.sol, config.PXN);
    auto sim_res = std::make_shared<Solver::SimResult>();
    sim_res->schedule = std::make_shared<Solver::EventSolution>();
    sim_res->schedule->init_devs = config.sol->init_devs;
    sim_res->schedule->chunk_sizes = config.sol->chunk_sizes;
    topo->set_simres(sim_res);

    LOOOG(PERFMODEL, TRACE, "Route: %s", route->toStr().c_str());
    auto &chunk_sizes = config.sol->chunk_sizes;
    for (auto root : route->roots) {
        auto src_c = root->chunk;
        std::set<Solver::SrcChunk> data = {src_c};
        auto chunk_size_B = chunk_sizes[src_c];
        int id = num_chunks++;
#ifdef USE_PIPELINED_MODEL
        int nslices = (chunk_size_B + SLICE_SIZE_B - 1) / SLICE_SIZE_B;
        for (int i = 0; i < nslices; ++i) {
            auto slice_size = std::min(SLICE_SIZE_B, chunk_size_B - i * SLICE_SIZE_B);
            auto chunk = std::make_unique<Chunk>(slice_size, route, root, data, src_c, i, nslices, id, 0, -1, -1);
            topo->send(std::move(chunk));
        }
#else
        auto chunk = std::make_unique<Chunk>(chunk_size_B, route, root, data, src_c, 0, 1, id, 0);
        topo->send(std::move(chunk));
#endif
    }

    // Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    for (auto npu : topo->npus) {
        auto device = npu.device;
        auto &recv = device->received_data;
        std::sort(recv.begin(), recv.end());
        int gpuid = device->get_local_id();
        sim_res->received_chunks[gpuid] = {};
        for (auto &data : recv) {
            sim_res->received_chunks[gpuid].push_back(data);
        }
    }
    const auto finish_time = event_queue->get_current_time();
    sim_res->time_us = (double)finish_time / 1000.0;
    LOOOG(PERFMODEL, TRACE, sim_res->toStr().c_str());
    return sim_res;
}
