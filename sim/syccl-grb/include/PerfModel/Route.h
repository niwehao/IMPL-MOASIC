/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Type.h"
#include "PerfModel/Topology.h"
#include "Solver/common.h"
#include <memory>
#include <map>
#include <string>

namespace Simulation {
     struct SendInstruction {
          // smaller means higher priority
          double prio;
          std::shared_ptr<RouteNode> next;
          send_type_e type;
          SendInstruction(double prio, std::shared_ptr<RouteNode> next, send_type_e type) : prio(prio), next(next), type(type) {}
     };

     class RouteNode {
     public:
          Solver::SrcChunk chunk;
          std::shared_ptr<Device> device;
          std::vector<SendInstruction> next_nodes;
          std::set<Solver::SrcChunk> reduced_data;
          double max_rem_t;

          RouteNode(Solver::SrcChunk chunk, std::shared_ptr<Device> device) : chunk(chunk), device(device) {}
          void connectNextNode(SendInstruction &next);
          std::string toStr(bool verbose = true);
     };

     // A tree of send algorithm
     class Route {
     private:
          void dfsRouteDepth(std::shared_ptr<RouteNode> cur, std::map<std::shared_ptr<RouteNode>, bool> &visited, ChunkSize chunk_size);
     public:
          std::shared_ptr<Topology> topology;
          std::vector<std::shared_ptr<RouteNode>> roots;
          Route(std::shared_ptr<Topology> topo, std::shared_ptr<Solver::EventSolution> sol, bool PXN);
          std::string toStr();
     };

     std::shared_ptr<Simulation::Link> getLink(std::shared_ptr<RouteNode> src, std::shared_ptr<RouteNode> dest);

}  // namespace Simulation
