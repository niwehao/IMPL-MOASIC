# Copyright (c) 2025, Alibaba Cloud and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json

def allgather_coll(chunk_size_B):
  coll = {}
  coll["name"] = "allgather"
  coll["byte"] = chunk_size_B
  coll["root_sender"] = -1
  coll["root_receiver"] = -1
  return coll

def alltoall_coll(chunk_size_B):
  coll = {}
  coll["name"] = "alltoall"
  coll["byte"] = chunk_size_B
  coll["root_sender"] = -1
  coll["root_receiver"] = -1
  return coll

def nvswitch_hosts(nhosts, ngpus, nnics):
  hosts = {}
  hosts["host_num"] = nhosts
  hosts["host_gpu_num"] = ngpus
  hosts["host_nic_num"] = nnics
  hosts["host_links"] = "nvswitch"
  return hosts

def multirail_topo(nrails):
  l0 = {"layer_id": 0, "type": "local", "link_spec": "memcpy"}
  l1 = {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
  l2 = {"layer_id": 2, "type": "nic", "link_spec": "link_nic"}
  l3 = {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": nrails, "link_spec": "netlink"}
  return [l0, l1, l2, l3]

def dgx2_topo():
  l0 = {"layer_id": 0, "type": "local", "link_spec": "memcpy"}
  l1 = {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
  l2 = {"layer_id": 2, "type": "nic", "link_spec": "pcie"}
  l3 = {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": 1, "link_spec": "netlink"}
  return [l0, l1, l2, l3]

def clos_topo(nleaf, nspine):
  l0 = {"layer_id": 0, "type": "local", "link_spec": "memcpy"}
  l1 = {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
  if nleaf == 0:
    return [l0, l1]
  l2 = {"layer_id": 2, "type": "nic", "link_spec": "link_nic"}
  l3 = {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": nleaf, "link_spec": "netlink_leaf"}
  if nspine == 0:
    return [l0, l1, l2, l3]
  l4 = {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": nspine, "link_spec": "netlink_spine"}
  return [l0, l1, l2, l3, l4]

def h800_1node_topo(nleaf):
  l0 = {"layer_id": 0, "type": "local", "link_spec": "memcpy"}
  l1 = {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
  l2 = {"layer_id": 2, "type": "nic", "link_spec": "link_nic"}
  l3 = {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": nleaf, "link_spec": "netlink"}
  return [l0, l1, l2, l3]

def sw_topo():
  l0 = {"layer_id": 0, "type": "local", "link_spec": "memcpy"}
  l1 = {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
  return [l0, l1]

def host_links():
  torus = {"dim_num": 2, "per_dim_num": [4, 4]}
  nvlink = [
    [1, 2, 3, 5],
    [0, 2, 3, 4],
    [0, 1, 3, 7],
    [0, 1, 2, 6],
    [5, 6, 7, 1],
    [4, 6, 7, 0],
    [4, 5, 7, 3],
    [4, 5, 6, 2]
  ]
  links = {}
  links["torus"] = torus
  links["nvlink"] = nvlink
  return links

def a100_link_specs(nhosts):
  specs = {}
  specs["memcpy"] = {"bw_mbpus": 10, "lat_us": 0.01}
  specs["nvlink"] = {"bw_mbpus": 0.30, "lat_us": 13.5}
  specs["link_nic"] = {"bw_mbpus": 0.0225, "lat_us": 0}
  specs["netlink_leaf"] = {"bw_mbpus": 0.0225, "lat_us": 25.5}
  specs["netlink_spine"] = {"bw_mbpus": 0.0225 * nhosts, "lat_us": 26.5}
  return specs

def a100_link_specs_small(nhosts):
  specs = {}
  specs["memcpy"] = {"bw_mbpus": 10, "lat_us": 0.01}
  specs["nvlink"] = {"bw_mbpus": 0.30, "lat_us": 4.5}
  specs["link_nic"] = {"bw_mbpus": 0.0225, "lat_us": 0}
  specs["netlink_leaf"] = {"bw_mbpus": 0.0225, "lat_us": 3}
  specs["netlink_spine"] = {"bw_mbpus": 0.0225 * nhosts, "lat_us": 3}
  return specs

def h800_link_specs():
  specs = {}
  specs["memcpy"] = {"bw_mbpus": 10, "lat_us": 0.01}
  specs["nvlink"] = {"bw_mbpus": 0.15, "lat_us": 10.5}
  specs["link_nic"] = {"bw_mbpus": 0.0455, "lat_us": 0}
  specs["netlink"] = {"bw_mbpus": 0.0455, "lat_us": 21.5}
  return specs

def h800_link_specs_small():
  specs = {}
  specs["memcpy"] = {"bw_mbpus": 10, "lat_us": 0.01}
  specs["nvlink"] = {"bw_mbpus": 0.15, "lat_us": 3}
  specs["link_nic"] = {"bw_mbpus": 0.0455, "lat_us": 0}
  specs["netlink"] = {"bw_mbpus": 0.0455, "lat_us": 10}
  return specs

def h800_link_specs_p2p():
  specs = {}
  specs["memcpy"] = {"bw_mbpus": 10, "lat_us": 0.01}
  specs["nvlink"] = {"bw_mbpus": 0.15, "lat_us": 0.35}
  specs["link_nic"] = {"bw_mbpus": 0.0455, "lat_us": 0}
  specs["netlink"] = {"bw_mbpus": 0.0455, "lat_us": 1.1}
  return specs

def mip_config():
  config = {
    "feasibility_tol": 1e-4,
    "intfeas_tol": 1e-4,
    "optimality_tol": 1e-4,
    "mip_gap": 0.001,
    "mip_focus": 1,
    "heuristics": 0.25,
    "time_limit_h": 5,
  }
  return config

def solver_config(epoch_time_strat, epoch_model, balance_ratio, alpha_epoch_duration_ratio_max):
  config = {
    "split_chunks": 1,
    "chunk_size_B": 0,
    "sw_hyperedge": False,
    "nic_hyperedge": True,
    "alpha_threshold": 0.1,
    "trial_time_factor": 1.0,
    "epoch_time_strat": epoch_time_strat,
    "epoch_model": epoch_model,
    "model_extra_sends_big_beta": False,
    "model_extra_sends_small_beta": False,
    "balance_ratio": balance_ratio,
    "alpha_epoch_duration_ratio_max": alpha_epoch_duration_ratio_max,
    "mip_config": mip_config()
  }
  return config

def convert_to_bytes(size_str):
    size_str = size_str.lower()
    if size_str.endswith("kb"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("mb"):
        return int(size_str[:-2]) * 1024**2
    elif size_str.endswith("gb"):
        return int(size_str[:-2]) * 1024**3

def sketch_config(sketch_output):
  config = {
    "customize_sketch": False,
    "use_sketch_input": False,
    "sketch_path": sketch_output
  }
  return config

def prune_config_ablation(prune_type, machine,
  iso_comb_prune, group_consistency_prune, start_source_diff_layer, 
  num_steps):
  config = {
    "max_comb_layer_num": 2,
    "allow_unequal_source_per_group": False,
    "ignored_layers": [
      0,
      2
    ],
    "must_contain_layers": [
      1,
      3
    ],
    "drop_small_sources": False,
    "drop_small_groups": False,
    "limit_num_steps": num_steps,
    "limit_num_steps_ratio": -1,
    "partition_sources": iso_comb_prune,
    "partition_dests": iso_comb_prune,
    "prune_dests": group_consistency_prune,
    "prune_sources_by_compare_steps": group_consistency_prune,
    "start_source_diff_layer": start_source_diff_layer,
    "layer_comm_ordering": {},
    "layer_comm_uplimit": {},
    "all_groups_used": False,
    "allow_source_group_special":True
  }
  return config

def prune_config_double(prune_type, machine):
  config = {
    "max_comb_layer_num": 2,
    "allow_unequal_source_per_group": False,
    "ignored_layers": [
      0,
      2
    ],
    "must_contain_layers": [
    ],
    "drop_small_sources": False,
    "drop_small_groups": False,
    "partition_sources": True,
    "partition_dests": True,
    "prune_dests": False,
    "prune_sources_by_compare_steps": False,
    "layer_comm_ordering": {},
    "layer_comm_uplimit": {},
    "all_groups_used": False,
    "allow_source_group_special":True
  }
  return config

def prune_config(prune_type, machine, algo_config={}):
  config = {
    "max_comb_layer_num": 3,
    "allow_unequal_source_per_group": False,
    "ignored_layers": [0, 2],
    "must_contain_layers": [1,3,4],
    "layer_comm_ordering": {},
    "layer_comm_uplimit": {"4": 16},
    "all_groups_used": False,
    "limit_num_steps": 3,
    "limit_num_steps_ratio": -1,
    "allow_source_group_special": True,
    "partition_sources": True,
    "partition_dests": True,
    "prune_dests": True,
    "prune_sources_by_compare_steps": True,
  }
  if machine == "a100":
    config["ignored_layers"] = [0, 2]
    assert len(algo_config["topo"]) in [2, 4, 5]
    if len(algo_config["topo"]) == 4:
        config["must_contain_layers"] = [1,3]
    else:
        config["must_contain_layers"] = [1,3,4]
    if prune_type == "small":
      config["layer_comm_uplimit"] = {}
      config["limit_num_steps"] = 5
      config["limit_num_steps_ratio"] = -1
      config["layer_comm_ordering"] = {}
      config["all_groups_used"] = False
    elif prune_type == "large":
      assert False
  elif machine == "h800":
    config["ignored_layers"] = [0,2]
    config["must_contain_layers"] = [1,3]
    config["layer_comm_ordering"] = {}
    config["layer_comm_uplimit"] = {}
    config["max_comb_layer_num"] = 2
    if prune_type == "small":
      config["limit_num_steps"] = 3
      config["limit_num_steps_ratio"] = -1
      config["layer_comm_ordering"] = {}
      config["all_groups_used"] = False
    elif prune_type == "large":
      assert False
  return config

def algo_solve_config(solve_output, filter_br, solve_br):
  config = {
    "filter_balance_ratio": filter_br,
    "solve_balance_ratio": solve_br,
    "raw_solve_balance_ratio": 1,
    "do_raw_solve": False,
    "num_used_algos_min": 3,
    "num_used_algos_max": 8,
    "time_diff_threshold": 0.2,
    "solve_output": solve_output,
  }
  return config

class ConfigGen:
  def __init__(self):
    self.CS_THRES = 1048576
    self.FILTER_BR = 3.0
    self.SOLVE_BR = 0.2
    self.EM = "ACCURATE"
    self.ES = "BETA_MULT"
  
  def set_filter_br(self, filter_br):
    self.FILTER_BR = filter_br
  
  def set_solve_br(self, solve_br):
    print(f"Setting solve_br to {solve_br}")
    self.SOLVE_BR = solve_br

  def a100conf(self, chunk_size_B, nhosts, ngpus, nnics, nleaf, nspine, prune_type, solve_output, sketch_output, a2a):
    config = {
      "coll": allgather_coll(chunk_size_B) if not a2a else alltoall_coll(chunk_size_B),
      "hosts": nvswitch_hosts(nhosts, ngpus, nnics),
      "topo": clos_topo(nleaf, nspine),
      "host_links": host_links(),
      "link_spec": a100_link_specs(nnics * nhosts) if chunk_size_B > self.CS_THRES else a100_link_specs_small(nnics * nhosts),
      "solver": solver_config(self.ES, self.EM, 1.0, 100),
      "sketch": sketch_config(sketch_output),
      "algo_solve": algo_solve_config(solve_output, self.FILTER_BR, self.SOLVE_BR)
    }
    config["prune"] = prune_config(prune_type, "a100", config)
    return config

  def h800conf(self, chunk_size_B, nhosts, ngpus, nnics, nrails, prune_type, solve_output, sketch_output, a2a):
    config = {
      "coll": allgather_coll(chunk_size_B) if not a2a else alltoall_coll(chunk_size_B),
      "hosts": nvswitch_hosts(nhosts, ngpus, nnics),
      "topo": multirail_topo(nrails),
      "host_links": host_links(),
      "link_spec": h800_link_specs() if chunk_size_B > self.CS_THRES else h800_link_specs_small(),
      "solver": solver_config(self.ES, self.EM, 1.0, 100),
      "sketch": sketch_config(sketch_output),
      "prune": prune_config(prune_type, "h800"),
      "algo_solve": algo_solve_config(solve_output, self.FILTER_BR, self.SOLVE_BR)
    }
    return config

  def ablconf(self, chunk_size_B, nhosts, ngpus, nnics, nrails, solve_output, sketch_output, a2a, iso_comb_prune, group_consistency_prune, start_source_diff_layer, filter_br, solve_br, nsteps):
    config = {
      "coll": allgather_coll(chunk_size_B) if not a2a else alltoall_coll(chunk_size_B),
      "hosts": nvswitch_hosts(nhosts, ngpus, nnics),
      "topo": multirail_topo(nrails),
      "host_links": host_links(),
      "link_spec": h800_link_specs() if chunk_size_B > self.CS_THRES else h800_link_specs_small(),
      "solver": solver_config("BETA_MULT", "ACCURATE", 1.0, 100),
      "sketch": sketch_config(sketch_output),
      "prune": prune_config_ablation("ablation", "h800", iso_comb_prune, group_consistency_prune, start_source_diff_layer, nsteps),
      "algo_solve": algo_solve_config(solve_output, filter_br, solve_br)
    }
    return config