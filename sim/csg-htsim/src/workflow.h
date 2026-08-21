// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef WORKFLOW_H
#define WORKFLOW_H

#include "config.h"

#include <fstream>
#include <sstream>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <vector>

struct SmartWorkflowTransfer {
    SmartWorkflowTransfer(uint32_t transfer_round, uint32_t src_node,
                          uint32_t dst_node, mem_b transfer_bytes)
        : round(transfer_round), src(src_node), dst(dst_node),
          bytes(transfer_bytes) {
    }

    uint32_t round;
    uint32_t src;
    uint32_t dst;
    mem_b bytes;
};

struct SmartTrafficMatrix {
    SmartTrafficMatrix() : node_count(0) {
    }

    SmartTrafficMatrix(uint32_t nodes)
        : node_count(nodes), bytes((size_t)nodes * nodes, 0) {
    }

    mem_b get(uint32_t src, uint32_t dst) const {
        return bytes[(size_t)src * node_count + dst];
    }

    void set(uint32_t src, uint32_t dst, mem_b value) {
        bytes[(size_t)src * node_count + dst] = value;
    }

    uint32_t node_count;
    std::vector<mem_b> bytes;
};

struct SmartWorkflow {
    SmartWorkflow() : node_count(0) {
    }

    SmartWorkflow(uint32_t nodes, uint32_t round_count)
        : node_count(nodes), rounds(round_count, SmartTrafficMatrix(nodes)) {
    }

    uint32_t round_count() const {
        return (uint32_t)rounds.size();
    }

    std::vector<SmartWorkflowTransfer> transfers_for_round(uint32_t round) const {
        std::vector<SmartWorkflowTransfer> transfers;
        if (round >= rounds.size()) {
            return transfers;
        }

        for (uint32_t src = 0; src < node_count; src++) {
            for (uint32_t dst = 0; dst < node_count; dst++) {
                mem_b bytes = rounds[round].get(src, dst);
                if (src != dst && bytes > 0) {
                    transfers.push_back(SmartWorkflowTransfer(round, src, dst, bytes));
                }
            }
        }
        return transfers;
    }

    uint32_t node_count;
    std::vector<SmartTrafficMatrix> rounds;
};

#include "../workload/A2A.h"

static SmartWorkflow build_workflow(const std::string& workload_name,
                                    uint32_t node_count,
                                    uint32_t round_count,
                                    bool all_to_all,
                                    uint32_t src,
                                    uint32_t dst,
                                    mem_b bytes_per_transfer) {
    if (workload_name == "A2A") {
        return A2AWorkload::build(node_count, round_count, all_to_all, src,
                                  dst, bytes_per_transfer);
    }
    throw std::runtime_error("unknown workload: " + workload_name);
}

// Demand file: '# comment' lines, one 'node_count N' line, then
// 'demand src dst bytes' lines (src != dst, bytes > 0). Used for
// non-uniform alltoallv workloads shared across schedulers.
static std::vector<std::vector<mem_b> > load_demand_matrix(
        const std::string& filename, uint32_t expected_nodes) {
    std::ifstream input(filename.c_str());
    if (!input) {
        throw std::runtime_error("cannot open demand file: " + filename);
    }

    uint32_t node_count = 0;
    std::vector<std::vector<mem_b> > demand;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string kind;
        stream >> kind;
        if (kind == "node_count") {
            stream >> node_count;
            if (node_count != expected_nodes) {
                throw std::runtime_error(
                    "demand file node_count does not match -nodes");
            }
            demand.assign(node_count, std::vector<mem_b>(node_count, 0));
        } else if (kind == "demand") {
            uint32_t src = 0, dst = 0;
            mem_b bytes = 0;
            stream >> src >> dst >> bytes;
            if (!stream || node_count == 0 || src >= node_count
                || dst >= node_count || src == dst || bytes <= 0) {
                throw std::runtime_error("bad demand line: " + line);
            }
            demand[src][dst] += bytes;
        } else {
            throw std::runtime_error("unknown demand file line: " + line);
        }
    }
    if (node_count == 0) {
        throw std::runtime_error("demand file missing node_count");
    }
    return demand;
}

static SmartWorkflow build_workflow_from_demand(
        const std::vector<std::vector<mem_b> >& demand, uint32_t node_count,
        uint32_t round_count) {
    SmartWorkflow workflow(node_count, round_count);
    for (uint32_t round = 0; round < round_count; round++) {
        for (uint32_t src = 0; src < node_count; src++) {
            for (uint32_t dst = 0; dst < node_count; dst++) {
                if (src != dst && demand[src][dst] > 0) {
                    workflow.rounds[round].set(src, dst, demand[src][dst]);
                }
            }
        }
    }
    return workflow;
}

static SmartWorkflow build_all_to_all_output_buffer_workflow(
        uint32_t node_count, uint32_t round_count,
        mem_b output_buffer_size) {
    if (node_count <= 1) {
        throw std::runtime_error("output-buffer A2A requires at least 2 nodes");
    }
    SmartWorkflow workflow(node_count, round_count);
    mem_b bytes_per_pair = (output_buffer_size + (node_count - 1) / 2)
        / (node_count - 1);
    for (uint32_t round = 0; round < round_count; round++) {
        for (uint32_t src = 0; src < node_count; src++) {
            for (uint32_t dst = 0; dst < node_count; dst++) {
                if (src != dst) {
                    workflow.rounds[round].set(src, dst, bytes_per_pair);
                }
            }
        }
    }
    return workflow;
}

#endif
