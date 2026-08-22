# Baseline:SyCCL(分层 fabric 搜索,MILP/LP + sketch 枚举)

> 代码结合版总结(2026-07-12)。solver 在 `syccl-grb/`(第三方,勿改算法)。

## 一句话

分层拓扑 DSL(local/host/nic/switch 层栈)上的求解器,三家中**求解最快**
(40s@32)且原生看得见 taper,但每层单一速率、交换机 one-big-switch
理想化、机内部分网实现不可用。

## 代码位置

| 组件 | 文件 | 说明 |
|---|---|---|
| config(2MB 基准) | `syccl-grb/config/rail-4h8g-a2a-2mb.json` | 层栈 + `link_spec`(**bw_mbpus 单位 = MB/µs**:0.3=2400G、0.045=360G、0.01125=90G、0.0225=180G);flex/pod 变体改 nvlink 层或 host_num |
| padded 变体 | `script/common.sh make_padded_syccl_config` | 改 `coll.byte` 与 `solve_output` |
| 求解 | `(cd syccl-grb/build && ./synthesize -f <config> solve)` | 结果 json 落在 build/ 下 `solve_output` 指定名 |
| json→.sched 转换 | `tools/syccl_generate_schedule.py` | `--explicit-path-topology dgx_rail`;**v5 修了拆分 chunk bug(见"坑"1)** |
| `.sched` 解析/执行 | `scheduler/syccl_schedule.h:74/389` | event 模式(可带 DAG);**非 DAG 模式 epoch 推进要求本 epoch 全部交付**(:496,唯一带交付屏障的 runtime);选包 = pool-order 轮转(:246,实验确定对它更优) |

## 优点

- 求解最快(40.5s@32,262s@64——唯一 64 可解的);
- 原生 taper 感知:spine 层 link_spec 调低即见收敛,dgx_rail 上无提示
  发现 rail-relay;
- 输出多个候选算法(挑最优)。

## 限制(报告 §1.4 SyCCL 行)

- **每层单一速率** ⇒ 异构混布(400/200/100G)不能表达(ndv2_16 出局);
- **机内部分网(`host_links:"nvlink"`)实现不可用**:上游两个 bug——
  越界写(PerfModel.cpp devtoGPU 缺失,已修)+ LP 解提取
  `DfsRateAlloc assert(fnd)` 失败(`syccl-grb/src/Solver/Solver.cpp:215`,
  **未修**,改它=改算法);只能 nvswitch 全交叉近似(ndv2_32 成绩
  "可执行非忠实");
- one-big-switch:端口级争用不可见(clos_32 的 224 个中继是真实负载);
- spine 层实践上单脊(connect_switches 非 pod 分组直接 assert)。

## 坑(必读)

1. **弱 NV 解会把一个 pair 拆成多个 chunk(addr = dst + k×ngpus)**。
   旧 converter 只按基础 chunk 记 demand → 这类解在 htsim 里
   delivered=0 直接 FAILED(连均匀都过不了)。v5 修复:demand 按最终
   chunk 表逐条生成(`tools/syccl_generate_schedule.py`,"One app-level
   demand per chunk" 注释处)。**新 nv 点的解如果 FAILED 且
   delivered_demands=0,先怀疑这一类问题。**
2. 结果 json 名 = config 里 `solve_output`(相对 build/);批处理脚本里
   glob 时留意命名(v5 曾因此空手);**必须命名为 result-*.json**——syccl-grb/.gitignore 只白名单放行 result- 前缀,否则解不会进版本库。
3. `coll.byte` 是每 chunk 字节;A2A 时 chunk 数 = ngpus-1/每 GPU。

## 怎么跑(复用优先)

```bash
NONUNIFORM=1 NONUNI_ALPHA=1.0 SCHEDULERS=syccl ./script/run_dgx_rail.sh
# 已有解:syccl-grb/build/result-{rail-4h8g,flex360,flex90}-*.json
#   与 cache/nonuniform/syccl_{dgx_rail,flex360,flex90}_a2a_*.sched
```

历史成绩(zipf1.0/seed42):nv2400=15927 / 360=15716.6 / 90=23231.1;
α∈{0.5,1.5,2.0}@2400 = 6779/27685/38147;n64 求解 262s(18 组)。
