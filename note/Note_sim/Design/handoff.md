# HANDOFF — MOSAIC/htsim 项目交接文档(Fable → Opus 4.8,2026-07-12)

写给接手的 agent:这份文档 + `CLAUDE.md`(铁律)是你的两份必读。CLAUDE.md
是每次都生效的硬约束;本文档是操作手册与坑清单,**按需查阅对应小节即可,
不必整读**。所有小节自成一体。

## 0. 项目一句话与当前状态

htsim 包级仿真上公平对比 CCL 调度方案(static_ecmp/basic/TE-CCL/TACCL/
SyCCL/FAST/MOSAIC)。**当前状态(2026-08-05,41 组)**:主报告
`Note/report_mosaic_eps.md`(v7,现行 SOTA = §11 **MOSAIC-U2 账本核**);统一方案
机制全自动门、不按 regime 换(现行 SOTA 设计与数字 = EPS 主干
§11;退役与负结果 → `Note/explore/` 各篇);41 组实验全部 validation PASSED 并归档
(`csg-htsim/experiments/mosaic_x1/`)。**主干同步铁律见 CLAUDE.md**
(SOTA 刷新 ⇒ 主干 eps 报告同步)。

## 1. 文件地图(先扫这张表,再按需打开)

| 你要找什么 | 去哪 |
|---|---|
| 硬性约定/铁律 | `CLAUDE.md`(仓库根) |
| 模拟器使用手册(编译/跑法/基准闸) | `Note/Design/README.md` |
| 主报告(动机→设计→数学→25 组实验→局限) | `Note/report_mosaic_eps.md` |
| D2 捐赠/skew 优先序探索(26/27 组;E1–E5 flag 全默认关,MOSAIC-P=skew_epoch2+rewaterfill) | `Note/explore/2026-0727-x26-27-donation-phase.md` |
| corridor ramp + 固定 wave size 探索(28 组;F1/F2/F3 默认关,MOSAIC-R=P+ramp_cold1+ramp_event,双自动门) | `Note/explore/2026-0727-x28-wave-ramp.md` |
| 机内 cube-mesh + NVLink 异构(29 组;新拓扑 flex_cube、MOSAIC WIRING_CUBE、M2 边界实测;cube 推荐 C6+W16) | `Note/explore/2026-0727-x29-flex-cube.md` |
| MOSAIC-LP(30 组;固定拓扑=机内 LP、不固定 workload=λ 调整;cube 推荐 C6+W12+lp+lp_local) | `Note/explore/2026-0727-x30-mosaic-lp.md` |
| MOSAIC-LP v2(31 组;FAST 差距归因=落点集中;hot-column jdist LP 统一摊布,cube 推荐再加 -mosaic_lp_spread jh 表) | `Note/explore/2026-0727-x31-mosaic-lp-v2.md` |
| 双悖论+冷源限速(32 组;23 组重跑结案;F6 -mosaic_cold_pace 三门;F7 -mosaic_host_lambda;explicit-spine 可达性门;仪器 -flex_degrade_dir/gpus) | `Note/explore/2026-0728-x32-paradox-pacing.md` |
| FAST 运行时 stage 同步(33 组;`-fast_sync` none/credit/barrier;`-fast_stage_order` desc 修正;保真度审计;FAST@2400 async 基线 3541→3419) | `Note/explore/2026-0728-x33-fast-runtime-sync.md` |
| 34 组:O1 波内配对着色(M1 翻盘,`-mosaic_pair_color_auto`)+ O3 λ 守卫(负结果) | `Note/explore/2026-0731-x34-pair-color.md`、`experiments/mosaic_x1/34_pair_color/` |
| **41 组:MOSAIC-L(资源账本核,EPS 立足点审计裁决)**(`-mosaic_ledger`;λ/三补丁/floor 退役 → 账本注水 + NIC 域 quorum 检测器;LU 7/8 ≥ SOTA、降速 2.3-2.6×、@90 残余 −5%;6 轮迭代) | `Note/explore/2026-0805-x41a-eps-foundation-audit.md`(审计)、`Note/explore/2026-0805-x41b-eps-ledger.md`(裁决)、`experiments/mosaic_x1/41_ledger/` |
| **44 组:MOSAIC-U2b(账本双通道广播协议)**(`-mosaic_lg_bcast -mosaic_lg_blag_us 2 -mosaic_lg_stagger_us 3`,兜底 `-mosaic_lg_spineagg`,诊断 `-mosaic_lg_forkprobe/-blagw`;重放证伪、错峰、leaf 聚合;已 promote sota_u2) | `Note/report_mosaic_eps.md` §11(v7.2)、`Note/explore/2026-0808-x44-lg-bcast.md`、`bench_runs/u2bl.csv` |
| 36 组:超大规模 weak scaling 256-1024(**chunk pool list+桶索引优化(全库逐位回归)**;FAST per-A2A 构造 111ms→2.70s vs MOSAIC 20ms;FAST/ECMP 默认 buffer 打爆) | `Note/explore/2026-0801-x36-scale-1024.md` |
| 37 组:switch-centric(纯 clos)incast/oversub/collision(MOSAIC min-buffer=**C/W** wave 定律 vs blast 0.9C;PC 均匀 −1~3%/skew 贴界;`tools/gen_incast.py`、`tools/clos_bounds.py`、clos buffer 旋钮) | `Note/explore/2026-0801-x37-switch-centric.md` |
| **38 组:统一方案 MOSAIC-U 定稿**(35 cell 全域扫:10 新 SOTA/8 持平/10 代价 ≤4.8%;PC 三重自动门(diag∧rail2p∧lp_spread);R/HOSTL 退役) | `Note/report_mosaic_eps.md` §10(历史;现行见 §11) |
| **现行 SOTA 配置与数字** | `Note/report_mosaic_eps.md` §11(主干同步铁律;SOTA.md 已删除) |
| 每个 baseline 的代码位置/优点/限制/坑 | `Note/Design/baseline_{teccl,taccl,syccl,fast,ecmp_basic}.md` |
| 非 SOTA 方案与失败搜索 | `Note/explore/` 各篇(fail.md 已删除,负结果直接写进对应 explore 报告) |
| FAST 论文精读 | `Note/Design/paper_fast_summary.md` |
| solver 能表达什么拓扑(全景表) | `Note/explore/2026-0708-solver-topology-capabilities.md` |
| 四 solver × 五拓扑旧结果全景 | `Note/explore/2026-0710-experiment-summary-v4.md` |
| 实验数据与图 | `csg-htsim/experiments/mosaic_x1/<NN_名字>/`(索引见其 README.md) |
| solver 产物/demand 矩阵/运行日志 | `csg-htsim/cache/nonuniform/`(x13/x15/x18/x21/x22/x23/x24 日志) |
| 调度器实现 | `csg-htsim/scheduler/*.h`;MOSAIC = `mosaic_schedule.h` |
| 拓扑实现 | `csg-htsim/topo/*.h`(flex_rail/flex_pod/flex_rail2p 都在 `flex_pod_graph.h`) |
| 工具 | `csg-htsim/tools/`(demand 生成、fill、下界、solver 转换) |

## 2. 结果复用规则(省钱省时,最重要的一节)

**只要满足以下全部条件,直接引用历史数字,不要重跑:**
1. baseline 的参数与设计没改(代码 diff 为空);
2. 拓扑参数相同(nodes/nv/nic/taper/hosts/gpus/spines/degrade 全同);
3. demand 相同(同文件或同 α/seed/mode);
4. 历史数字来自 validation PASSED(experiments/*/results.csv 里都是)。

查数顺序:`experiments/mosaic_x1/<NN>/results.csv` → 没有再查
`cache/nonuniform/x*.log` → 都没有才跑。**主线 8 个 nv 点
(demand = zipf1.0/seed42)的权威数字表 = `21_design_iter/results.csv`
(每行含 bound/FAST/MOSAIC-C4/MOSAIC-C6 列)——引用前扫一眼该组
README 确认前提。行文注意:"第 NN 组"指目录编号(如第 21 组 =
21_design_iter),与"共 25 组实验"的总数是两回事。solver 的 `.sched` 解也一样:
文件在 cache 里就直接 fill+执行,**不要重新求解**(TE-CCL 生成器本身
内容寻址缓存;TACCL/SyCCL 靠文件存在检查;`FORCE_RESOLVE=1` 才强制重解)。

**必须重跑的情况**:改了 mosaic/fast 的调度代码(哪怕一行)、改了拓扑
builder、换了 demand。改 MOSAIC 后的最小回归集见 §6。

## 3. 实验操作手册(step-by-step)

### 3.0 新方案对比 = 基准闸一键(2026-08-05 起,优先于手工逐 cell)

```bash
NAME=myscheme SCHED=mosaic_schedule FLAGS="-mosaic_ledger -my_flag" \
  bash script/bench_scheme.sh
# 67 cell 全跑(主线/种子/α/persrc/uniform/khot/尺寸/形态/n64/pod/
# 降速/taper/陈旧先验)→ 自动与 bench_baselines.csv 拼表:
#   experiments/mosaic_x1/bench_runs/<NAME>.csv + stdout 逐 cell 倍数
#   (vs sota_u2 / vs 最强其它)+ 晋升门 PASS/BREACH 判定
# baseline 永不重跑;新方案要成为未来对照:
python3 tools/bench_compare.py myscheme --promote
```

### 3.1 编译
```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
```
改了任何 .h 都要重编(全 header-only)。编译错误会直接打印;没有 error
字样即成功。

### 3.2 跑一个在线方案(MOSAIC/FAST/ECMP/basic)
```bash
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail \
  -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 \
  -scheduler mosaic_schedule -mosaic_corridors 6 \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
```
- 取结果:输出里 `validation PASSED ... completion_us X` 一行。
  **FAILED 的数字禁止进对比表**(CLAUDE.md 铁律)。
- `-chunk_size` 非均匀时 = demand 文件头部的 `padded_pow2_bytes`
  (zipf1.0=33554432,zipf2.0=67108864);均匀时 = 2097152 并加
  `-message_size 2097152`(去掉 -demand_file)。
- FAST 把 `-scheduler mosaic_schedule -mosaic_corridors 6` 换成
  `-scheduler fast_schedule -fast_alpha_beta_guard`(guard 必带,公平性)。
- ECMP/basic:`-scheduler static_ecmp` / `basic`,无额外旋钮。

### 3.3 参数速查(常用旋钮;完整见报告 §9)

| 旋钮 | 含义 | 默认/建议 |
|---|---|---|
| `-flex_nv_gbps` | 每 GPU scale-up 带宽(NV:NIC 分子) | 2400;扫 {2400,720,600,480,420,360,180,90} |
| `-flex_nic_gbps` | 每 GPU NIC | 180(全部实验恒定) |
| `-flex_taper_gbps` | rail:每 leaf 上行;pod:每 leaf 上行**总额** | 180(rail)/1440·720·360(pod) |
| `-flex_hosts/-flex_gpus_per_host` | 形态 | 4×8;nodes 必须等于乘积 |
| `-flex_spines` | spine 数(rail 分摊 taper;pod 同) | 1(rail)/4(pod) |
| `-flex_degrade_gpu G -flex_degrade_factor F` | 单 GPU NIC 双向降为 nic/F,调度器不知情 | 关 |
| `-flex_planeb_factor F` | 仅 flex_rail2p:B 面全 fabric ÷F | 1 |
| `-mosaic_corridors C` | corridor 数(rail:1..6 嵌套;pod:1+lane 数;rail2p:5=三管族) | **新实验:rail 传 6,pod/rail2p 传 5**。代码默认 4 仅为逐位复现 v4 历史数字,不是推荐值 |
| `-mosaic_waves/-mosaic_gamma/-mosaic_floor/-mosaic_threshold` | W/γ/ε/波阈值 | 6/0.7/0.02/0.75(全场统一,勿逐点调) |
| `-mosaic_static/-mosaic_uniform_init/-mosaic_rx_init/-mosaic_prior_demand_file` | 消融:关反馈/关先验/朴素初值/陈旧先验 | 关 |
| `-mosaic_explicit_spine` | pod:显式 per-spine lane;**规则:G%S==0 用隐式(不加),否则加** | 关 |
| `-mosaic_known_planeb F` | rail2p:静态异构先验(K-管注水用) | 1 |
| `-fast_rr_select/-fast_scaling B` | FAST 公平性敏感检查 | 关(03 组已做) |
| `-ecn_threshold_bytes/-pause_threshold_bytes` | **0=不覆盖队列内建默认**(不是关闭!全方案同底座) | 0 |

### 3.4 跑 solver(TE-CCL/TACCL/SyCCL)
最简路径(dgx_rail ≡ flex_rail@nv2400,自动缓存/填充/执行):
```bash
NONUNIFORM=1 NONUNI_ALPHA=1.0 SCHEDULERS="teccl taccl syccl" ./script/run_dgx_rail.sh
```
非 2400 的 nv 点没有现成脚本,流程 = 造该 nv 的 solver 侧拓扑描述 → 求解
→ 转换 → fill → 执行;逐家的具体命令、文件名约定、耗时预期都写在
`Note/Design/baseline_{teccl,taccl,syccl}.md` 的"怎么跑"小节——**照抄
那里的命令即可**。时间预算:TE-CCL 2-15min、TACCL 11-17min、SyCCL 1-5min
(32 GPU);>30min 记 SKIPPED+原因,这本身是合法结论。

### 3.5 生成 demand / 算下界
```bash
python3 tools/gen_nonuniform_a2a.py --nodes 32 --message-bytes 2097152 \
  --alpha 1.0 --seed 42 --output cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# 模式:--mode {hotdst(默认)|persrc|khot --khot K}
# 注意:hotdst 的每 dst 机内:跨机比是常数 → 先验对 seed/α 不变,
#   drift/先验类实验必须用 persrc(报告 §6.5 方法学注记)。
python3 tools/x1_bounds.py --demand <文件> --nv-gbps 360 \
  [--gpus-per-host 8] [--degrade-gpu G --degrade-factor F] \
  [--leaf-uplink-total T]   # pod 用;输出 bound_us 与 binding 项
```

### 3.6 归档一组新实验(必须做,CLAUDE.md 铁律)
1. `mkdir experiments/mosaic_x1/<NN_名字>`(NN 递增,现在到 25);
2. 写 `results.csv`(逐列命名清楚,附 bound_us 列)与 `README.md`
   (中文:设置、主表、结论、复现命令;**对我们不利的数据也要如实写**);
3. 在 `experiments/mosaic_x1/plot_all.py` 末尾照抄既有段落风格加一段
   `if _has("<NN_名字>"):`,跑
   `/home/weihaoni/miniconda3/envs/myenv/bin/python3 plot_all.py`;
4. 更新 `experiments/mosaic_x1/README.md` 索引表一行;
5. 若刷新 SOTA:按 CLAUDE.md 主干同步铁律更新 `Note/report_mosaic_eps.md`(§11)+ 基准闸缓存(--promote);若进报告:更新对应小节,
   **逐数与 CSV 核对后再写**。

### 3.7 长批次的正确跑法(Opus 注意)
- 写一个批处理 .sh(参考 `/tmp/...scratchpad` 里的历史脚本模式:每 run
  一行 `echo "TAG k=v ... $(grep validation)" >> LOG`),用
  run_in_background 跑**一层**后台(**不要在脚本里再 `&` 二层后台——
  父 shell 退出会把孤儿杀掉,这个坑踩过**);
- 结束标记(如 `echo DONE >> LOG`)+ 完成后解析 LOG 归档;
- 一次改一个变量;跑之前先查 §2 能不能复用。

## 4. 未修 bug / 刻意保留的行为(改动前必读)

| # | 事项 | 状态与原因 |
|---|---|---|
| 1 | TE-CCL 上游内置拓扑 9 个 bug(mesh 边界、amd off-by-one 等) | **未修**(铁律:不改 solver);清单在 `Design/baseline_teccl.md`;规避 = 只用 RegularGraph 入口 |
| 2 | SyCCL `DfsRateAlloc assert(fnd)`(机内部分网 LP 解提取崩溃) | **未修**(算法内部);规避 = nvswitch 全交叉近似,并标注"可执行非忠实" |
| 3 | TACCL 时间哨兵(routing 5e6/scheduler 5e5)+ 小数 α 停滞 | **未修**;规避 = `make_padded_taccl_topo` 的时间单位重标定+α 取整,勿手搓 |
| 4 | clos_32 上 TE-CCL 慢 17%(min-epoch 映射空转) | **刻意保留**:换轮转会伤 TACCL(+59%);选包语义按 solver 分治(SyCCL 轮转、其余 min-epoch),勿全局改 |
| 5 | `-ecn_threshold_bytes 0` 语义 = 不覆盖内建默认(`src/wait_queue.h:33`) | 不是 bug 但反直觉:极端 backlog 下会看到 ECN 计数,全方案一致、合法 |
| 6 | mosaic 代码默认 `-mosaic_corridors 4` | **刻意**:保证 01-12 组 v4 数字逐位可复现;新实验手动传 6(报告 §6.9 是 v5 建议默认) |
| 7 | smart_regular.dat(仓库根) | 无用的 Logfile 空壳(numrecords=0),每次运行重生成;可忽略或改 main_regular.cpp:609 路径 |
| 8 | teccl hop 模式 fill 后的 id 重编号 | fill 工具已做;自写转换器时记得(runtime 要求连续 id) |
| 9 | SyCCL 拆分 chunk(addr=dst+k·N) | converter 已修(v5);症状 = delivered_demands=0 的 FAILED |

## 5. 各方案一句话画像(细节点开 Design/)

- **static_ecmp**:铁 baseline,1× 基准;路径钉死的代价量尺。
- **basic**:直连稀疏图之王(regular_32 全场最快),flex 系列上 ≈ECMP。
- **TE-CCL**:表达力最强、求解最慢;必须 ONE_SHOT;64 GPU 超时。
- **TACCL**:机内画像强、fabric 盲;sketch 要人工;必须 cd 到 cache 跑。
- **SyCCL**:求解最快、每层单速;弱 NV 解会拆 chunk;非 DAG 模式带交付屏障。
- **FAST**:最强在线对手;guard 必带;W1-W7 + 模运算结构盲(偶 S 半 spine)。
- **MOSAIC**:我方;缺点如实列在报告 §7.1(M1-M7),别藏。

## 6. 改 MOSAIC 设计后的最小回归集(依次跑,全 PASSED 才算过)

1. 向后兼容:C=4 默认在 (nv360, zipf1.0/seed42) 应逐位复现 **8565.57**;
2. 主线 8 nv 点 × C=6(对照 **第 21 组** `experiments/mosaic_x1/21_design_iter/results.csv` 的 mosaic_c6_us 列,劣化 >1% 即回归);
3. 稳健性抽查:seed43@360、zipf1.5@90、persrc@360、uniform@2400;
4. 场景抽查:taper90@360、degrade26×4@2400、pod(360,720)、rail2p B/4@2400;
5. 每次都看 `queue_drops`(应 0)与 hop_bytes 膨胀是否异常。
以上历史值全在 `experiments/mosaic_x1/{21,11,19,22,24}_*/results.csv`。

## 7. 用户的工作要求(除 CLAUDE.md 外的软约定,来自协作历史)

- 回答/文档默认简体中文,关键术语英文;引用结论给文件:行号;
- **诚实优先**:对我们不利的结果如实进报告(§7.1 M 项就是范例);
  审稿人视角自审:审出问题改文档/补实验,**不重复已 PASSED 的实验**;
- 设计改进以"运行时相对 FAST 的领先"为标准(生成速度已经足够快);
- subagent 同时 ≤4,实验并行优先自己管后台进程;
- 大任务先写计划拆小步,每步有可验证的产出;上下文吃紧先写 handoff。

## 8. 如果要继续研究,按优先级(报告 §8 的展开)

1. E0-b oracle:TE-CCL 直接解真实非均匀矩阵,标定中段 bound gap;
2. 跨 collective λ warm-start(M5 的解法);
3. M1 翻盘:波内配对 permutation coloring(跨 dst 协调);
4. 真实系统落地路径见报告 §8.3(TCCL 的三条工程路径)。
