# SyCCL → htsim Schedule 合成流程指南

面向后续模型/工程师的 runbook:给定 csg-htsim 里的一种拓扑,如何用 SyCCL
(本目录 gurobi 版,二进制 `build/synthesize`)求解生成 schedule 并在 htsim
中执行对比。以 `dgx_rail`(32 GPU rail-optimized)为已验证的工作示例。

## 0. 前提与不变约束

- `build/synthesize` 已编译(重编译需 GUROBI_HOME + FindGUROBI.cmake);
  Gurobi 学术 license 有效。
- **默认 workload 契约**:A2A,每对 2MB(`coll.byte=2097152`,SyCCL 的
  chunk 直接 = coll.byte),包 1KB,htsim `-store_forward`。
- 求解时限:`solver.mip_config.time_limit_h ≤ 0.15`(9 分钟)。
- 不修改求解器源码;只写 config json + adapter。

## 1. 写 config(SyCCL 的拓扑是"分层"描述)

参考已验证的 `config/rail-4h8g-a2a-2mb.json`。要点:

- `hosts`:`host_num`/`host_gpu_num`/`host_nic_num`/`host_links:"nvswitch"`。
  GPU i → NIC ⌊i/(gpu_per_nic)⌋;每 GPU 独立 NIC 就设 host_nic_num=host_gpu_num。
- `topo` 分层数组,每层 `type` + `link_spec` 键:
  - layer0 `local`(memcpy),layer1 `host`(NVSwitch 全交叉),layer2 `nic`;
  - **leaf 层**:`{"type":"switch","switch_topo":"multirail","switch_num":R}`
    —— `multirail` = 按 NIC 序号跨主机分组(轨道布线);`pod` = 按整机连续分组。
    `switch_num == host_nic_num` 时每轨恰好一台 leaf。
  - **spine 层**:`{"type":"switch","switch_topo":"pod","switch_num":1}`
    (脊层代码只支持 pod,实践上单脊;taper 通过给 `netlink_spine` 低带宽表达)。
- `link_spec`:`bw_mbpus` 单位 MB/us(180Gbps≈0.0225),`lat_us` 微秒。
  数值必须与 htsim 拓扑一致(dgx_rail:nvlink 0.3/9,nic 0.0225/0,
  leaf 0.0225/25,spine 0.0225/25=4:1 taper)。
- `prune`:`must_contain_layers` 要包含 spine 层号,否则搜索空间排除它
  (dgx_rail 用 [1,3,4],ignored [0,2]);`layer_comm_uplimit` 控制规模。
- `algo_solve.solve_output`:结果 JSON 路径(相对 build/ 运行目录)。

## 2. 求解

```bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/syccl-grb/build
./synthesize -f ../config/rail-4h8g-a2a-2mb.json solve
```
- dgx_rail 实测:40.5 秒,输出 3 个候选 algorithm(按预测完成时间升序),
  algorithms[0].final_schedule.Time = 预测完成时间(us)。
- 输出 JSON 在 `algo_solve.solve_output` 指定的位置(相对 build/)。

## 3. 转换为 htsim .sched

```bash
cd /home/weihaoni/Tage/JUMP-sim/Htsim/csg-htsim
python3 tools/syccl_generate_schedule.py \
  --syccl-result ../syccl-grb/build/result-rail-4h8g-a2a-2mb.json \
  --explicit-path-topology dgx_rail \
  --output cache/syccl_dgx_rail_a2a_2mb.sched
```
- 默认取 algorithms[0](最快),`--algorithm-index` 可换;
  runtime 模式保持默认 `syccl_event_dag`(数据流 DAG,勿用 `syccl_event`)。
- 新拓扑需在该工具加 `--explicit-path-topology <name>` 分支:
  switch-as-wire 拓扑下 path 一律 `[src_gpu, dst_gpu]`。
- adapter 内置校验(新拓扑也要保留):A2A 的 `amount` 必须 ≡1.0
  (reconstruct 后逐跳整块中继语义)、`reduce` 全 False、
  统计 layer_used 分布与跨轨 send 数;出现"同主机但 layer_used≥2"
  (想走 NIC)的 send 要警告——htsim 同主机只有 NVLink 逻辑边。

## 4. htsim 执行与验收

```bash
./sim/tests/htsim_smart_regular -nodes 32 -topology dgx_rail -packet_size 1024 \
  -scheduler syccl_schedule -schedule_file cache/syccl_dgx_rail_a2a_2mb.sched \
  -workload A2A -all_to_all -store_forward -quiet
```
验收:`validation PASSED`,`expected_bytes 2080374784`,`queue_drops 0`;
对照 solver 预测(final.Time):dgx_rail 实测 2377.41us vs 预测 2272.2us(+4.6%),
偏差 >10% 时先怀疑 config 与 htsim 拓扑数值不一致。
参考:vs static_ecmp 7852.29us 加速 3.30×;layer 分布 928×NVLink + 768×leaf,
spine=0(SyCCL 无提示自主发现 rail-relay)。

## 5. 已知陷阱

1. `host_links:"nvswitch"` 时 config 里的 `host_links.nvlink` 邻接矩阵**不生效**
   (仅 host_type=="nvlink" 才读),别被现成 config 里那份矩阵误导。
2. SyCCL 的 epoch 是全局离散步并携带每链路每 epoch 容量;htsim 执行是
   数据可用性驱动 + epoch 作优先级(runtime 刻意如此,预测偏差 <5% 证明可接受)。
3. `send_dict_*.npy` / `cs_ts_cr_tr_*.pkl` 是 TACCL/TECCL 侧产物,SyCCL 的
   输出只有 result JSON,别混淆。
4. 网络内部争用在 SyCCL 模型里是"one-big-switch"理想化(taper 只体现在
   spine 链路 β 上);htsim 是它的更真实评判者,两者数字定义不同属正常。
5. `csg-htsim/tools/syccl_generate_config.py` 是旧的简化 clos 生成器,
   **不能**生成本流程的 rail config;直接手写/复制 `config/rail-*.json` 改。

## 6. ndv2_32 实战补充(2026-07-08)

6. **`host_links:"nvlink"` 邻接矩阵路径不可用**(出厂 config 从未走过该路径,
   两个上游 bug 叠加):
   - bug#1 越界写:`PerfModel.cpp` connect_gpus_by_nvlink 用全局 DeviceId 直接
     索引按 GPU 序号排布的 `npus`,堆损坏 → malloc_consolidate abort。
     **已于 2026-07-08 修复**(补 devtoGPU 转换 + 断言,见源码 BUGFIX 注释)。
   - bug#2 解提取断言:修复 bug#1 后暴露,`Solver.cpp:216
     extractLPEventSolution` 里 `DfsRateAlloc` 找不到流路径,`assert(fnd)`
     失败——LP 解→事件链回构算法对主机内部分网的假设不成立。属求解算法
     内部问题,未修。
   结论不变:要表达主机内部分网(如 NDv2 cube-mesh)仍只能降级:config 用
   `nvswitch` 全交叉近似,adapter 侧把非邻接的主机内 send 沿真实邻接展开成
   2 跳(htsim 物理侧忠实,solver 会高估这些 send 的独享带宽)。此时该拓扑
   对 SyCCL 属"可执行非忠实",对比结论要加脚注。
7. **`layer_comm_uplimit` 别加在唯一的跨主机层上**:会让有限步数搜索全灭
   (A2A 直发一步需要 32 路并发),而 `limit_num_steps:-1` 时会搜出病态深
   中继算法(实测比 ECMP 慢 4 倍)。单交换层拓扑用 `layer_comm_uplimit {}`
   + `limit_num_steps 2`。ndv2_32 参考 config:`config/ndv2-32-a2a-2mb.json`。
