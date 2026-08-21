#include "Sketch/Collective.h"
#include <cassert>
#include <thread>

const char* coll_str[] = {
        "SendRecv",
        "Reduce", "Gather",
        "Broadcast", "Scatter",
        "AllReduce", "AllGather", "Alltoall", "ReduceScatter",
        "Sub-Collective", "Group-Collective", "Extend-Group-Collective"
};

Collective::Collective(std::string coll_name, int ngpus, int byte, int sender_gpu, int receiver_gpu):
        root_collective(nullptr), reduce(false), ngpus(ngpus) {
  // All collective schedules can be generated from the below primitives
  if (coll_name == "broadcast") {
    coll_type = coll_broadcast;
    for (int r = 0; r < ngpus; r++) {
      chunk_t sender_chunk = {sender_gpu, 0, byte, true};
      chunk_t receiver_chunk = {r, 0, byte, false};
      sender_chunks[sender_gpu][sender_chunk].insert(receiver_chunk);
      receiver_chunks[r][receiver_chunk].insert(sender_chunk);
    }
  } else if (coll_name == "scatter") {
    coll_type = coll_scatter;
    for (int r = 0; r < ngpus; r++) {
      chunk_t sender_chunk = {sender_gpu, r, byte, true};
      chunk_t receiver_chunk = {r, 0, byte, false};
      sender_chunks[sender_gpu][sender_chunk].insert(receiver_chunk);
      receiver_chunks[r][receiver_chunk].insert(sender_chunk);
    }
  }
    // Many-to-many
  else if (coll_name == "allgather") {
    coll_type = coll_allgather;
    for (int s = 0; s < ngpus; s++) {
      for (int r = 0; r < ngpus; r++) {
        chunk_t sender_chunk = {s, 0, byte, true};
        chunk_t receiver_chunk = {r, s, byte, false};
        sender_chunks[s][sender_chunk].insert(receiver_chunk);
        receiver_chunks[r][receiver_chunk].insert(sender_chunk);
      }
    }
  } else if (coll_name == "alltoall") {
    coll_type = coll_alltoall;
    for (int s = 0; s < ngpus; s++) {
      for (int r = 0; r < ngpus; r++) {
        chunk_t sender_chunk = {s, r, byte, true};
        chunk_t receiver_chunk = {r, s, byte, false};
        sender_chunks[s][sender_chunk].insert(receiver_chunk);
        receiver_chunks[r][receiver_chunk].insert(sender_chunk);
      }
    }
  } else {
    assert(0);
  }
  finish();
}

Collective::Collective(Collective* coll, std::set<chunk_t>& chunk_set) :
        root_collective(coll), coll_type(coll->coll_type), reduce(coll->reduce), ngpus(coll->ngpus) {
  for (auto c : chunk_set) {
    if (coll->sender_chunks.find(c.gpu_id) != coll->sender_chunks.end() &&
        coll->sender_chunks[c.gpu_id].find(c) != coll->sender_chunks[c.gpu_id].end()) {
      sender_chunks[c.gpu_id][c] = coll->sender_chunks[c.gpu_id][c];
    }
    if (coll->receiver_chunks.find(c.gpu_id) != coll->receiver_chunks.end() &&
        coll->receiver_chunks[c.gpu_id].find(c) != coll->receiver_chunks[c.gpu_id].end()) {
      receiver_chunks[c.gpu_id][c] = coll->receiver_chunks[c.gpu_id][c];
    }
  }
  finish();
}

void Collective::finish() {
  for (auto it : sender_chunks)
    sender_gpus.push_back(it.first);
  for (auto it : receiver_chunks)
    reciever_gpus.push_back(it.first);
}

chunk_t Collective::getChunk(int gpu_id, bool is_sender) {
  if (is_sender) {
    assert(sender_chunks.find(gpu_id) != sender_chunks.end());
    assert(sender_chunks[gpu_id].size() == 1);
    return sender_chunks[gpu_id].begin()->first;
  } else {
    assert(receiver_chunks.find(gpu_id) != receiver_chunks.end());
    assert(receiver_chunks[gpu_id].size() == 1);
    return receiver_chunks[gpu_id].begin()->first;
  }
}

std::string Collective::toStr() {
  std::string msg = std::string(coll_str[coll_type]);
  msg += ", " + std::to_string(ngpus) + "gpu";

  msg += ", \nsend=[";
  for (const auto& it: sender_chunks) {
    for (auto sender_chunk : it.second) {
      msg += std::to_string(sender_chunk.first.gpu_id) + "." + std::to_string(sender_chunk.first.chunk_id) + ",";
    }
    msg.pop_back();
  }
  msg += "]";

  msg += ", \nrecv=[";
  for (const auto& it: receiver_chunks) {
    for (const auto& recv_chunk : it.second) {
      msg += std::to_string(recv_chunk.first.gpu_id) + "." + std::to_string(recv_chunk.first.chunk_id) + "(";
      for (auto send_chunk : recv_chunk.second) {
        msg += std::to_string(send_chunk.gpu_id) + "." + std::to_string(send_chunk.chunk_id) + ",";
      }
      msg.pop_back();
      msg += "),";
    }
    msg.pop_back();
  }
  msg += "]";
  return msg;
}
