// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PerfModel/Device.h"
#include "PerfModel/Link.h"

namespace Simulation {
    /*
     * Network layer and group information
     */

    enum layer_group_type {
        group_local = 0,
        group_switch, // GPUs inside a group is connected by a "switch"
        group_ring,       // GPUs inside a group is connected by a ring
        group_direct      // GPUs inside a group is connected by specified p2p links
    };

    enum layer_relationship_type {
        layer_repeat = 0,
        layer_contain,
        layer_be_contained,
        layer_cross
    };

    struct network_group_t {
        int layer_id;
        int group_id;
        layer_group_type type;

        std::set<std::shared_ptr<Device>> devices;
        std::shared_ptr<Device> switch_device = nullptr;

        std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>> links;
        std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::vector<std::shared_ptr<Link>>>> paths;

        std::set<int> connected_gpus;
        // global gpu id -> local group id
        std::map<int, int> global_to_local_ids;
        std::map<int, int> local_to_global_ids;
        std::map<std::shared_ptr<Device>, std::set<std::shared_ptr<Device>>> device_to_connected_gpus;

        std::map<int, std::vector<int>> contained_groups;
        std::string toStr();
    };

    struct layer_info_t {
        std::vector<network_group_t> groups;
        std::map<int, layer_relationship_type> layer_topo_relation;
    };

    struct topo_info_t {
        std::vector<layer_info_t> layer_info;
        std::map<std::shared_ptr<Device>, std::map<int, int>> device_groups;
        std::map<std::shared_ptr<Device>, std::pair<int, int>> switch_device_groups;
        int get_layer_num_links(int layer);
        Bandwidth get_layer_total_bandwidth(int layer);
        Bandwidth get_layer_avg_bandwidth(int layer);
        std::string toStr();
    };
}