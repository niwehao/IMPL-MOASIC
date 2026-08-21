# TACCL → htsim Schedule 合成流程指南

面向后续模型/工程师的 runbook:给定 csg-htsim 里的一种拓扑,如何用 TACCL
求解生成 schedule 并在 htsim 中执行对比。以 `dgx_rail`(32 GPU rail-optimized)
为已验证的工作示例。

## 0. 前提与不变约束

- conda env `myenv`(taccl CLI 可用),Gurobi 学术 license 有效。
- **默认 workload 契约**:A2A,每对 2MB,包 1KB,htsim `-store_forward`。
- **时限**:两阶段 MIP 各 ≤600s。scheduler 阶段用环境变量
  `TACCL_SCHEDULER_TIME_LIMIT=600`;routing 阶段已改为读
  `TACCL_ROUTING_TIME_LIMIT`(默认 600,见 `taccl/routing.py:40`,
  原为硬编码 1800——这是求解参数调整,算法未动)。
- 不修改求解算法;只写 topo profile / sketch / adapter。

## 1. 写 topology profile(物理画像)

TACCL 的拓扑输入是"性能画像"而不是图:`gpus_per_node` / `nics_per_node` /
`alpha` / `betas` / `invbws` / `remote_alpha` / `remote_beta` / `remote_invbw`。
参考已验证的 `csg-htsim/examples/dgx_rail/taccl_dgx_rail_topo_2mb.json`:

- 单位是 0.1us;数值按**目标 chunk 大小**(2MB)标定:
  - NVLink:alpha 90(9us)、beta 67(2MB@300GB/s≈6.67us)、invbw=alpha+beta=157
  - internode(NIC 180G):remote_alpha 500(50us=25+25 两段)、
    remote_beta 932(2MB@22.5GB/s≈93.2us)、remote_invbw 1432
- `nics_per_node` 影响 beta_split_factor(=gpus/nics);每 GPU 独立 NIC 时为 1,
  remote_beta 不被缩放。
- **换 chunk 大小必须重标定 beta/invbw**(它们内含传输时间)。

## 2. 写 sketch(人工结构提示——TACCL 方法论的一部分)

参考 `csg-htsim/examples/dgx_rail/taccl_dgx_rail_sketch.json`:
- `"nnodes": 4`,internode 策略 `"relay"`;
- **rail 对齐**:`"internode_conn": "direct-map"`(= 本地 GPU g 只连远端 GPU g,
  见 taccl/cli/common.py 对 direct-map 的解析)。这把跨主机 send 锁在同轨,
  轨对齐由主机内 NVLink 中继完成——正是 dgx_rail 上打败 ECMP 的关键结构;
- `"symmetry_offsets": [[8, 32]]`(4 主机旋转对称,数量级压缩 MIP 规模)。
  没有 relay+symmetry,32 GPU 的 A2A MIP 会爆炸。

## 3. 求解

```bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
TACCL_SCHEDULER_TIME_LIMIT=600 TACCL_ROUTING_TIME_LIMIT=600 \
conda run -n myenv taccl solve custom Alltoall \
  --topology-file examples/dgx_rail/taccl_dgx_rail_topo_2mb.json \
  --sketch-file examples/dgx_rail/taccl_dgx_rail_sketch.json \
  -d cache/taccl_dgx_rail -f
# 输出文件名带时间戳:取 ls -t cache/taccl_dgx_rail/*_taccl.sccl.json | head -1
```
- 注意 CLI 细节(实测):子命令拓扑名是小写 `custom`;`solve` 没有 `-o`,
  用 `-d DIR` 指定输出目录,文件名形如 `Alltoall.n32-...-<ts>_taccl.sccl.json`。
- 输出是 **solve JSON**(Algorithm 序列化),不是 msccl XML(不要走 ncclize)。
- dgx_rail 实测:routing MIP 4.4s OPTIMAL;scheduler MIP 600s 截止 gap 1.86%
  (可接受;要更优就提高时限)。
- 求解产生的大 `.lp`(百 MB 级)可删。

## 4. 转换为 htsim .sched

```bash
python3 tools/taccl_to_htsim_schedule.py \
  --input cache/taccl_dgx_rail/taccl_solution.json \
  --output cache/taccl_dgx_rail/taccl_dgx_rail_a2a_2mb.sched \
  --chunk-size-bytes 2097152 --topology dgx_rail
```
- 新拓扑需在该工具里加 `--topology <name>` 分支:switch-as-wire 拓扑下
  path 一律 `[src, dst]`(GPU 逻辑全连接);校验 rank 数;统计跨主机跨轨 send
  数(rail sketch 下应为 0)。
- **不要对 step 内 sends 重排**——顺序就是求解器的 t(发送时刻)顺序,
  htsim runtime 靠注入顺序保留它。

## 5. htsim 执行与验收

```bash
./sim/tests/htsim_smart_regular -nodes 32 -topology dgx_rail -packet_size 1024 \
  -scheduler taccl_schedule -schedule_file cache/taccl_dgx_rail/taccl_dgx_rail_a2a_2mb.sched \
  -workload A2A -all_to_all -store_forward -quiet
```
验收:`validation PASSED`,`expected_bytes 2080374784`(992 demand × 2MB),
`queue_drops 0`。dgx_rail 参考值:completion 2432.89us(vs ECMP 7852.29us,3.23×)。

## 6. 已知陷阱

1. TACCL 的 send 是**逻辑单跳 GPU→GPU**,其带宽/时延已由 profile 建模;
   htsim 侧逻辑边(交换机即线缆)与之一一对应,不要再展开成多跳物理路径
   (旧版 adapter 对 ndv2/a100 的硬编码展开就是双重计费的来源)。
2. htsim runtime 的 epoch 屏障已放宽为"注入即推进":唯一硬约束是
   chunk 可用性;epoch 只作发送优先级与节点内顺序。这忠于 TACCL
   "step 只是连续时间离散化"的原义。
3. solve JSON 里没有 instances/channel(那是 ncclize 阶段的);并行链路
   索引 l 在链路重数=1 的拓扑(dgx_rail)无影响,链路重数>1 的拓扑要先扩展
   adapter/runtime 支持再用。
4. sketch 决定成败:没有对称性约束和 relay 限制,32 GPU 规模求解会超时;
   internode_conn 的选择直接编码了你希望 schedule 呈现的跨机结构。
