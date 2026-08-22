# Baseline:TACCL(sketch 引导的两阶段 MIP)

> 代码结合版总结(2026-07-12)。solver 在 `taccl/`(第三方,勿改算法)。

## 一句话

"同构主机复制品"模型上的 routing→ordering 两阶段 MIP,靠人写的
communication sketch 剪搜索空间——**机内异构 NVLink 画像是看家本领,
主机间 fabric 完全不可见**。

## 代码位置

| 组件 | 文件 | 说明 |
|---|---|---|
| topo 画像(2MB 标定) | `csg-htsim/examples/dgx_rail/taccl_dgx_rail_topo_2mb.json` | alpha=90、intra betas=67、invbw=alpha+beta、remote_alpha/beta;flex 变体 = **只把 intra betas ×(2400/nv) 取整并同步 invbw=α+β,α 与 remote_alpha/beta 不动**(已验证:flex360 topo alpha=90/beta=447/remote 500·932 与基准一致;α 的重标定只发生在 padded 阶段、由 make_padded_taccl_topo 自动做)。样例在 `cache/nonuniform/taccl_flex*_topo_*.json` |
| padded 重标定 | `script/common.sh make_padded_taccl_topo` | **时间单位重标定 + α 取整**(见"坑"1) |
| sketch | `examples/dgx_rail/taccl_dgx_rail_sketch.json` | 换布线/规模要人工重写 |
| 求解调用 | `script/run_dgx_rail.sh` taccl 分支 | **必须 `cd` 到该拓扑的 cache 目录跑**(见"坑"2);时限 env:`TACCL_ROUTING_TIME_LIMIT`/`TACCL_SCHEDULER_TIME_LIMIT`(默认各 600s) |
| json→.sched 转换 | `tools/taccl_to_htsim_schedule.py` | `--topology dgx_rail --chunk-size-bytes <pad>`。**`--topology` 指布线结构名(节点编号/布线),flex 任意 nv 都填 dgx_rail**——带宽差异只活在 solver topo json 与 htsim 的 `-flex_nv_gbps` 里,布线不变 |
| `.sched` 解析/执行 | `scheduler/taccl_schedule.h:57/322` | chunk/demand/step 三段;死锁前瞻 `pending_can_eventually_be_satisfied`(:697);min-epoch 选包 |

## 优点

- 机内任意链路矩阵 + 每链路重数(DGX-1 式 1x/2x NVLink 混布);
- 中等 pad(8MB)下表现好:α=0.5 时 4915,是三家 solver 最好(15 组);
- 求解含 ordering(相对 TE-CCL 的纯流量解多一层调度语义)。

## 限制(报告 §1.4 TACCL 行)

- **主机间 fabric 不可见**:internode 只有一个标量链路类——taper/spine/
  pod 结构全靠 sketch 侧面约束(dgx_rail 是人工禁跨轨才像样);
- **sketch 税**:每换布线/规模要人工重设计(18 组:n=64 因此跳过);
  cube-mesh 机内必须收窄 2-relay 否则 routing MIP 零可行解;
- **S1/S2**:pad-to-max + 每 α 重解 ~686s(RUN15 实测);MIP 时限内留
  gap(600s,1.86%)。

## 坑(必读)

1. **硬编码时间哨兵**:routing ST=5e6 / scheduler ST=5e5。大 chunk 画像
   直接 β×16 会撞哨兵报 Infeasible(不是不可解);α 重标定成小数会让
   MIP 找不到可行解(1e-9 容差下停滞)。正确做法 = **时间单位重标定 +
   α 取整**,用 `make_padded_taccl_topo`,勿手搓。nv=90 的 flex 变体
   (beta×26.7)同样可解——哨兵问题只由编码方式引起。
2. **按 CWD 相对路径读写工作文件**(`taccl/taccl/routing.py:357,407`、
   `scheduler.py:461,557`:`model_sccl_*.lp/.ilp`、`cs_ts_cr_tr_*.pkl`、
   `send_dict_*.npy`,且 pkl/npy 会被后续阶段读回)——调用必须
   `(cd "$TACCL_DIR" && taccl solve ...)`,给工具传绝对路径。
3. conda 环境:`conda run -n myenv taccl ...`(Gurobi license 在该 env)。

## 怎么跑(复用优先)

```bash
NONUNIFORM=1 NONUNI_ALPHA=1.0 SCHEDULERS=taccl ./script/run_dgx_rail.sh
# 已有解:cache/nonuniform/taccl_dgx_rail_{8,32,64}mb/、taccl_flex{360,90}_32mb/
# 求解一次 ~11-17min(routing ≤600s + scheduler ≤600s + 转换)
```

历史成绩(zipf1.0/seed42):nv2400=17451 / 360=15302 / 90=31299;
α∈{0.5,1.5,2.0}@2400 = 4915/28076/38917(全 PASSED)。
