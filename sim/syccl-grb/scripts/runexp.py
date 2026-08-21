# Copyright (c) 2025, Alibaba Cloud and its affiliates
# SPDX-License-Identifier: Apache-2.0

from config_gen import *
import os, subprocess

def solve(config_path, solver_path, log_path, thrs=None):
  # return
  cmd = [solver_path, "-f", config_path, "solve", "|", "tee", log_path]
  if thrs:
    cmd = ["SYNTHESIZE_PARALLEL_THREAD_NUM=%d" % thrs] + cmd
  else:
    cmd = ["SYNTHESIZE_PARALLEL_THREAD_NUM=72"] + cmd
  with open("./exp_progress.txt", "a") as f:
    f.write(" ".join(cmd) + "\n")
  print(" ".join(cmd))
  subprocess.run(" ".join(cmd), shell=True)

def run_a100(nhosts, ngpus, nnics, nleaf, nspine, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= 4
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      prefix = f"{cs}B-prune={prune_type}"
      chunk_size_B = cs // (nhosts * ngpus)
      conf = cfg.a100conf(chunk_size_B, nhosts, ngpus, nnics, nleaf, nspine, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
      if not os.path.exists(os.path.join(dir, "configs")):
        os.mkdir(os.path.join(dir, "configs"))
      config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
      with open(config_path, "w") as f:
        f.write(json.dumps(conf, indent=2))
      solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4 # if nhosts <= 8 else 8
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  if nhosts >= 32:
    chunk_sizes = chunk_sizes[3:]
    chunk_sizes.append(chunk_sizes[-1] * 4)
    chunk_sizes.append(chunk_sizes[-1] * 4)
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      prefix = f"{cs}B-prune={prune_type}"
      chunk_size_B = cs // (nhosts * ngpus)
      conf = cfg.h800conf(chunk_size_B, nhosts, ngpus, nnics, nrails, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
      if not os.path.exists(os.path.join(dir, "configs")):
        os.mkdir(os.path.join(dir, "configs"))
      config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
      with open(config_path, "w") as f:
        f.write(json.dumps(conf, indent=2))
      solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800_p2p(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4 if nhosts <= 8 else 8
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      prefix = f"{cs}B-prune={prune_type}"
      chunk_size_B = cs // (nhosts * ngpus)
      conf = cfg.h800conf_p2p(chunk_size_B, nhosts, ngpus, nnics, nrails, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
      if not os.path.exists(os.path.join(dir, "configs")):
        os.mkdir(os.path.join(dir, "configs"))
      config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
      with open(config_path, "w") as f:
        f.write(json.dumps(conf, indent=2))
      solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_a1002(nhosts, ngpus, nnics, nleaf, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= 4
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      prefix = f"{cs}B-prune={prune_type}"
      chunk_size_B = cs // (nhosts * ngpus)
      conf = cfg.a100conf(chunk_size_B, nhosts, ngpus, nnics, nleaf, 0, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
      if not os.path.exists(os.path.join(dir, "configs")):
        os.mkdir(os.path.join(dir, "configs"))
      config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
      with open(config_path, "w") as f:
        f.write(json.dumps(conf, indent=2))
      solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800_ar(nhosts, ngpus, nnics, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      prefix = f"{cs}B-prune={prune_type}"
      chunk_size_B = cs // (nhosts * ngpus)
      conf = cfg.h800arconf(chunk_size_B, nhosts, ngpus, nnics, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
      if not os.path.exists(os.path.join(dir, "configs")):
        os.mkdir(os.path.join(dir, "configs"))
      config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
      with open(config_path, "w") as f:
        f.write(json.dumps(conf, indent=2))
      solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800_abl_no_diff_layer(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      for steps in [3, 5, 10]:
        for iso_comb_prune in [True, False]:
          for group_consistency_prune in [True, False]:
            start_source_diff_layer = False
            prefix = f"{cs}B-prune={prune_type}-iso_comb_prune={iso_comb_prune}-group_consistency_prune={group_consistency_prune}-start_source_diff_layer={start_source_diff_layer}-steps={steps}"
            chunk_size_B = cs // (nhosts * ngpus)
            fs_ratio = 1.0 if a2a else 0.2
            conf = cfg.ablconf(chunk_size_B, nhosts, ngpus, nnics, nrails, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a, iso_comb_prune, group_consistency_prune, start_source_diff_layer, 3.0, fs_ratio, steps)
            if not os.path.exists(os.path.join(dir, "configs")):
              os.mkdir(os.path.join(dir, "configs"))
            config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
            with open(config_path, "w") as f:
              f.write(json.dumps(conf, indent=2))
            solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))


def run_h800_abl_with_diff_layer(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      for steps in [3, 5, 10]:
        for iso_comb_prune in [True, False]:
          for group_consistency_prune in [True, False]:
            start_source_diff_layer = True
            prefix = f"{cs}B-prune={prune_type}-iso_comb_prune={iso_comb_prune}-group_consistency_prune={group_consistency_prune}-start_source_diff_layer={start_source_diff_layer}-steps={steps}"
            chunk_size_B = cs // (nhosts * ngpus)
            fs_ratio = 1.0 if a2a else 0.2
            conf = cfg.ablconf(chunk_size_B, nhosts, ngpus, nnics, nrails, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a, iso_comb_prune, group_consistency_prune, start_source_diff_layer, 3.0, fs_ratio, steps)
            if not os.path.exists(os.path.join(dir, "configs")):
              os.mkdir(os.path.join(dir, "configs"))
            config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
            with open(config_path, "w") as f:
              f.write(json.dumps(conf, indent=2))
            solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800_abl_ratio(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  for cs in chunk_sizes:
    for prune_type in ["small"]:
      for filter_br in [8.0, 3.0, 1.0]:
        for solve_br in [3.0, 1.0, 0.5, 0.2, 0.1]:
          if solve_br >= filter_br:
            continue
          prefix = f"{cs}B-prune={prune_type}-filter_br={filter_br}-solve_br={solve_br}"
          chunk_size_B = cs // (nhosts * ngpus)
          conf = cfg.ablconf(chunk_size_B, nhosts, ngpus, nnics, nrails, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a, True, True, False, filter_br, solve_br, 3)
          if not os.path.exists(os.path.join(dir, "configs")):
            os.mkdir(os.path.join(dir, "configs"))
          config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
          with open(config_path, "w") as f:
            f.write(json.dumps(conf, indent=2))
          solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"))

def run_h800_scale_test(nhosts, ngpus, nnics, nrails, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  ratio = 4 if nhosts <= 8 else 8
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= ratio
  thrs = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
  for cs in chunk_sizes:
    for thr in thrs:
      for prune_type in ["small"]:
        prefix = f"{cs}B-prune={prune_type}-thr={thr}"
        chunk_size_B = cs // (nhosts * ngpus)
        conf = cfg.h800conf(chunk_size_B, nhosts, ngpus, nnics, nrails, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
        if not os.path.exists(os.path.join(dir, "configs")):
          os.mkdir(os.path.join(dir, "configs"))
        config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
        with open(config_path, "w") as f:
          f.write(json.dumps(conf, indent=2))
        solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"), thrs=thr)

def run_a100_scale_test(nhosts, ngpus, nnics, nleaf, nspine, dir, a2a, cfg):
  dir = os.path.join(dir, "a2a" if a2a else "ag")
  chunk_sizes = []
  # total chunk size: 1KB to 4GB
  cs = 2 ** 10
  os.makedirs(dir, exist_ok=True)
  while cs <= 4 * (2 ** 32):
    chunk_sizes.append(cs)
    cs *= 4
  thrs = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
  for cs in chunk_sizes:
    for thr in thrs:
      for prune_type in ["small"]:
        prefix = f"{cs}B-prune={prune_type}-thr={thr}"
        chunk_size_B = cs // (nhosts * ngpus)
        conf = cfg.a100conf(chunk_size_B, nhosts, ngpus, nnics, nleaf, nspine, prune_type, os.path.join(dir, f"{prefix}-result.json"), os.path.join(dir, "configs", "sketch.json"), a2a)
        if not os.path.exists(os.path.join(dir, "configs")):
          os.mkdir(os.path.join(dir, "configs"))
        config_path = os.path.join(dir, "configs", f"{prefix}-config.json")
        with open(config_path, "w") as f:
          f.write(json.dumps(conf, indent=2))
        solve(config_path, "./synthesize", os.path.join(dir, f"{prefix}-log.txt"), thrs=thr)

def runnormal(name, od, br):
  cfg = ConfigGen()
  cfg.set_solve_br(float(br))
  for a2a in [False, True]:
    if name == "a100":
      run_a100(nhosts=4, ngpus=8, nnics=4, nleaf=2, nspine=1, dir=os.path.join(od, "a100-4hosts-8gpu-4nic-clos"), a2a=a2a, cfg=cfg)
    elif name == "a100-2":
      run_a1002(nhosts=2, ngpus=8, nnics=4, nleaf=1, dir=os.path.join(od, "a100-2hosts-8gpu-4nic-clos"), a2a=a2a, cfg=cfg)
    elif name == "h800":
      run_h800(nhosts=8, ngpus=8, nnics=8, nrails=8, dir=os.path.join(od, "h800-8hosts-8gpu-8nic-rail"), a2a=a2a, cfg=cfg)
    elif name == "h800-32":
      run_h800(nhosts=32, ngpus=8, nnics=8, nrails=8, dir=os.path.join(od, "h800-32hosts-8gpu-8nic-rail"), a2a=a2a, cfg=cfg)
    elif name == "h800-64":
      run_h800(nhosts=64, ngpus=8, nnics=8, nrails=8, dir=os.path.join(od, "h800-64hosts-8gpu-8nic-rail"), a2a=a2a, cfg=cfg)
    elif name == "h800-abl-nodiff":
      run_h800_abl_no_diff_layer(nhosts=6, ngpus=4, nnics=4, nrails=4, dir=os.path.join(od, "h800-8hosts-8gpu-8nic-rail-abl"), a2a=a2a, cfg=cfg)
    elif name == "h800-abl-ratio":
      run_h800_abl_ratio(nhosts=6, ngpus=4, nnics=4, nrails=4, dir=os.path.join(od, "h800-8hosts-8gpu-8nic-rail-abl"), a2a=a2a, cfg=cfg)
    elif name == "h800-abl-withdiff":
      run_h800_abl_with_diff_layer(nhosts=6, ngpus=4, nnics=4, nrails=4, dir=os.path.join(od, "h800-8hosts-8gpu-8nic-rail-abl"), a2a=a2a, cfg=cfg)
    elif name == "h800-scale-test":
      run_h800_scale_test(nhosts=8, ngpus=8, nnics=8, nrails=8, dir=os.path.join(od, "h800-8hosts-8gpu-8nic-rail-scale"), a2a=a2a, cfg=cfg)
    elif name == "a100-scale-test":
      run_a100_scale_test(nhosts=4, ngpus=8, nnics=4, nleaf=2, nspine=1, dir=os.path.join(od, "a100-4hosts-8gpu-4nic-clos-scale"), a2a=a2a, cfg=cfg)
    else:
      print(f"Unknown experiment {name}")

import sys
if __name__ == "__main__":
  name, od, br = sys.argv[1], sys.argv[2], sys.argv[3]
  runnormal(name, od, br)