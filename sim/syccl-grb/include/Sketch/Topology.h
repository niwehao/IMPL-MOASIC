// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

enum node_type_e {
    NODE_SWITCH_NET = 0,
    NODE_SWITCH_NV,
    NODE_GPU,
    NODE_NIC,
    NODE_NONE,
};

enum coll_prim_e {
    // One-to-one
    coll_sendrecv=0,
    // Mnay-to-one
    coll_reduce,
    coll_gather,
    // One-to-many
    coll_broadcast,
    coll_scatter,
    // Many-to-many
    coll_allreduce,
    coll_allgather,
    coll_alltoall,
    coll_reducescatter,
};