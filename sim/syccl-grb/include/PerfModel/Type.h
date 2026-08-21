/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "Solver/common.h"

class CommGraph;

namespace Simulation {
    class Device;
    class Chunk;
    class Link;
    class Route;
    class RouteNode;
    class Topology;

    using send_type_e = Solver::LinkType;
    inline send_type_e get_send_type(node_type_e src, node_type_e dst) {
        return Solver::getLinkType(src, dst);
    }
    inline bool sendCountBeta(send_type_e type) {
        return Solver::linkCountBeta(type);
    }

    /// Callback function pointer: "void func(void*)"
    using Callback = void (*)(void*);

    /// Callback function argument: void*
    using CallbackArg = void*;

    /// Device ID which starts from 0
    using DeviceId = int;

    /// Chunk size in Bytes
    using ChunkSize = uint64_t;

    /// Bandwidth in GB/s(as input, converted later to B/ns)
    using Bandwidth = double;

    /// Latency in ns
    using Latency = double;

    /// Inverse Bandwidth in ns/Byte
    using InvBandwidth = double;

    /// Event time in ns
    using EventTime = Solver::EventTime;

}  // namespace Simulation
