// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Solver/common.h"
#include <string>

namespace SolverVars {
  using ChunkVar = Solver::SrcChunk;
  using LinkVar = Solver::LinkInfoId;

  inline std::string Node(int node) {
    return "N" + std::to_string(node);
  }

  inline std::string Chunk(ChunkVar c) {
    auto [src, pos] = c;
    return Node(src) + "_" + std::to_string(pos);
  }

  inline std::string Link(LinkVar l) {
    auto [src, dst, id] = l;
    return Node(src) + "->" + Node(dst) + ":" + std::to_string(id);
  }

  inline std::string Epoch(int epoch) {
    return "E" + std::to_string(epoch);
  }

  inline std::string SendAtEpoch(ChunkVar c, LinkVar l, int epoch) {
    return "send " + Chunk(c) + " to " + Link(l) + "@" + Epoch(epoch);
  }

  inline std::string Buffer(ChunkVar c, int node, int epoch) {
    return "buffer " + Chunk(c) + "@" + Node(node) + "@" + Epoch(epoch);
  }

  inline std::string Consume(ChunkVar c, int node, int epoch) {
    return "consume " + Chunk(c) + "@" + Node(node) + "@" + Epoch(epoch);
  }

  inline std::string Received(ChunkVar c, int dest, int epoch) {
    return "received " + Chunk(c) + "@" + Node(dest) + " by " + Epoch(epoch);
  }
}