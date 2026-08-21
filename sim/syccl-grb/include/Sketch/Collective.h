// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sketch/Topology.h"
#include <map>
#include <set>
#include <vector>
#include <string>
#include <cassert>

extern const char* coll_str[];

struct chunk_t {
    int gpu_id;
    int chunk_id;
    int byte;
    bool is_sender;

    bool operator<(const chunk_t &other) const {
      assert(byte == other.byte);
      if (is_sender != other.is_sender) return is_sender < other.is_sender;
      if (gpu_id != other.gpu_id) return gpu_id < other.gpu_id;
      return chunk_id < other.chunk_id;
    }

    bool operator==(const chunk_t &other) const {
      assert(byte == other.byte);
      return is_sender&other.is_sender && gpu_id == other.gpu_id && chunk_id == other.chunk_id;
    }
};

class Collective {
public:
    Collective* root_collective;
    coll_prim_e coll_type;
    bool reduce;
    int ngpus;
    int byte;

    std::map<int, std::map<chunk_t, std::set<chunk_t>>> sender_chunks;
    std::map<int, std::map<chunk_t, std::set<chunk_t>>> receiver_chunks;
    std::vector<int> sender_gpus;
    std::vector<int> reciever_gpus;

    Collective(std::string coll_name, int ngpus, int byte, int sender_gpu, int receiver_gpu);
    Collective(Collective* coll, std::set<chunk_t>& chunks);

    chunk_t getChunk(int gpu_id, bool is_sender);
    void finish();

    std::string toStr();
};
