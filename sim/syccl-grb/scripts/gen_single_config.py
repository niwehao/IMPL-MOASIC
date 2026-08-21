# Copyright (c) 2025, Alibaba Cloud and its affiliates
# SPDX-License-Identifier: Apache-2.0

from config_gen import *

# generate the config file
# example of allgather, 4 machines * 8 gpus per machine, clos topology
chunk_size_B = 16777216
nhosts = 4
ngpus = 8
nnics = 4
nleaf = 2
nspine = 1
prune_type = "small"
output = "result-a100-4hosts-8gpu-4nic-clos.json"
a2a = False
balance_ratio = 0.5
cfg = ConfigGen()
cfg.set_solve_br(balance_ratio)
conf = cfg.a100conf(chunk_size_B, nhosts, ngpus, nnics, nleaf, nspine, prune_type, output, a2a)
config_path = "../config/a100-4hosts-8gpu-4nic-clos.json"
with open(config_path, "w") as f:
    f.write(json.dumps(conf, indent=2))

# example of alltoall, 8 machines * 8 gpus per machine, rail topology
chunk_size_B = 67108864
nhosts = 8
ngpus = 8
nnics = 8
nrails = 8
prune_type = "small"
output = "result-h800-8hosts-8gpu-8nic-rail.json"
a2a = True
balance_ratio = 0.5
cfg = ConfigGen()
cfg.set_solve_br(balance_ratio)
conf = cfg.h800conf(chunk_size_B, nhosts, ngpus, nnics, nrails, prune_type, output, a2a)
config_path = "../config/h800-8hosts-8gpu-8nic-rail.json"
with open(config_path, "w") as f:
    f.write(json.dumps(conf, indent=2))