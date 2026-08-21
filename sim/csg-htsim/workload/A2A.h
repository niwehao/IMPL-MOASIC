// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef A2A_WORKLOAD_H
#define A2A_WORKLOAD_H

#include "../src/workflow.h"

class A2AWorkload {
public:
    static SmartWorkflow build(uint32_t node_count,
                               uint32_t round_count,
                               bool all_to_all,
                               uint32_t src,
                               uint32_t dst,
                               mem_b bytes_per_transfer) {
        if (node_count == 0) {
            throw std::runtime_error(
                "workflow node count must be greater than zero");
        }
        if (round_count == 0) {
            throw std::runtime_error("workflow must have at least one round");
        }
        if (bytes_per_transfer <= 0) {
            throw std::runtime_error(
                "workflow transfer bytes must be greater than zero");
        }

        SmartWorkflow workflow(node_count, round_count);
        for (uint32_t round = 0; round < round_count; round++) {
            if (all_to_all) {
                install_all_to_all_round(workflow, round, bytes_per_transfer);
            } else {
                install_bidirectional_round(workflow, round, src, dst,
                                            bytes_per_transfer);
            }
        }
        return workflow;
    }

private:
    static void install_all_to_all_round(SmartWorkflow& workflow,
                                         uint32_t round,
                                         mem_b bytes_per_transfer) {
        for (uint32_t flow_src = 0; flow_src < workflow.node_count; flow_src++) {
            for (uint32_t flow_dst = 0; flow_dst < workflow.node_count;
                 flow_dst++) {
                if (flow_src != flow_dst) {
                    workflow.rounds[round].set(flow_src, flow_dst,
                                               bytes_per_transfer);
                }
            }
        }
    }

    static void install_bidirectional_round(SmartWorkflow& workflow,
                                            uint32_t round,
                                            uint32_t src,
                                            uint32_t dst,
                                            mem_b bytes_per_transfer) {
        workflow.rounds[round].set(src, dst, bytes_per_transfer);
        if (src != dst) {
            workflow.rounds[round].set(dst, src, bytes_per_transfer);
        }
    }
};

#endif
