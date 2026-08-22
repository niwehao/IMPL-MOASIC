# Baseline:FAST(在线 Birkhoff 分解调度器)——代码结合版

> 2026-07-12。论文精读见同目录 `paper_fast_summary.md`(本文不重复论文
> 内容,只写代码实现、实测行为、公平性旋钮与坑)。官方源码 `FAST/`
> (algorithm 逐行移植,勿改)。

## 一句话

启动时对任意(可非均匀)demand 矩阵现算三步计划:sender balance →
server 级矩阵 Birkhoff 置换分解 → receiver dispatch;产出 path 模式的
`SmartTecclSchedule`,复用 TE-CCL runtime 回放。**最强在线 baseline,
也是 MOSAIC 的直接对照轴。**

## 代码位置(全部在 `scheduler/fast_schedule.h`)

| 组件 | 行号 | 对照官方 |
|---|---|---|
| plan 构建入口 | `FastPlanBuilder::build()`:234 | — |
| sender balance | `balance_server_pair()`:383 | `FAST/simulation/scheduler/local.cpp` |
| α-β guard | `guard_skip_balance()`:439 | `local.cpp:187-216`(忠实移植) |
| Birkhoff 分解 | `decompose()`:526,匈牙利增广 `augment()`:585 | `algorithm/all2all.cpp` |
| dispatch | `pull_row()`:464 / `emit_chain()`:482 | — |
| epoch 布局 | balance=0、机内 A2A=1、stage p 的 inter=1+p、dispatch=2+p(头注 :14-18) | — |
| 选包 | min-epoch(:73;`-fast_rr_select` 切轮转) | — |
| stage 顺序 | `build()` 内 stable_sort 按 freq 降序(`-fast_stage_order plan` 回退提取序) | `nvidia/.../alltoall_global_scheduler.cc:287-292`(largest-first) |
| 运行时 stage 同步 | `teccl_schedule.h` `enable_fast_sync()`/`fast_gate_open()`(`-fast_sync none/credit/barrier`,默认 none) | `nvidia/.../flash_alltoall_nvshmem.cu`(credit)/ `simulation/scheduler/global.cpp:150-212`(barrier) |
| plan 计时 | main 打印 `fast_plan ... plan_us` | — |

## 运行时同步旋钮(33 组新增,详见 `Note/explore/2026-0728-x33-fast-runtime-sync.md`)

- `-fast_sync none`(默认):历史行为,chunk availability 是唯一硬约束,
  各 stage 异步并发——对 FAST **有利**的上界近似(无 straggler 等待)。
- `-fast_sync credit`:忠实移植官方 NVSHMEM kernel 的执行期门控
  (本 rank balance 收齐→首个 inter;per-rank stage 串行;接收端
  双缓冲 credit:收齐 p-1 且 dispatch 完 p-2 才授信 stage p;
  dispatch 等本 stage 收齐且排在 intrinsic/上一 dispatch 之后)。
- `-fast_sync barrier`:官方分析模型的全局 stage 屏障(段序列
  [bal]→[inter0‖intra]→[inter p‖disp p-1]→[disp P-1]),同步代价上界。
- `-fast_stage_order desc`(默认):官方 production 的 largest-freq-first
  stage 序;`plan` = 本移植旧行为(Birkhoff 提取序,复现 ≤32 组归档数字)。

## 优点(实测)

- 原生非均匀,plan 0.6ms@32(超线性:109ms@256,18 组);
- 强 scale-up + 均匀/温和 skew 是主场:uniform@2400 赢 MOSAIC 8~10%,
  persrc@2400 赢 5%(全局 permutation 一对一无 incast);
- 治接收端 incast:非均匀 2400 点贴 server 入口 cut +8.1%。

## 弱点(报告 §1.3 W1–W7,全部实测;此处只列坑相关)

- **W7 每字节两过机内层**:NV:NIC < ~4:1 起塌陷(结构模型
  `T_FAST ≈ (same+(1−1/G)cross)/βV`,实测偏差 ≤1.6%,报告 §4.9 Prop 4);
- **W1 fabric 盲**:taper 三档数字逐位相同(11 组);pod 4:1 超额时比
  ECMP 还慢(22 组);
- **模运算结构盲**(v5.1 新发现):i→i 同 index 中继 + 模钉死 ⇒
  双平面 `(s+d)%2` 下**永远只用 A 面**(23 组:三档 planeb 逐位不变、
  对称时只用一半 fabric);偶数 spine 数下**只用一半 spine**
  (`2i mod S` 恒偶,25 组:S=4 14629 vs S=3/5 10126)。

## 公平性义务(CLAUDE.md 铁律,报告 §5.1 第 4/5 条)

- **对比必须带 `-fast_alpha_beta_guard`**(官方 guard);
- 质疑其数字前先跑 `-fast_rr_select`(选包语义)与 `-fast_scaling <B>`
  (官方 scaling)敏感性(03 组已做:弱侧塌陷与这些无关,且默认
  min-epoch 对它更有利);
- `-fast_gpus_per_server` 缺省按拓扑自动(flex 系列 = flex_gpus_per_host)。

## 坑

1. 不接受 `-schedule_file`(在线现算,main 里直接拒绝);
2. expected_bytes 含中继 hop 字节(> app 字节),看 validation 的
   delivered 与 expected 一致即可;
3. 极端 backlog 下会触发**默认 ECN/pause 底座**(全方案一致,见
   handoff.md"ECN 语义");α=2.0@90 的 100459µs 含 49 次 rate cut,合法。

## 历史成绩速查(zipf1.0/seed42,PASSED)

nv 2400→90(8 点,async 按点两序取优,33 组更新):
**3419**/5068¹/5988/7369/8746/10126/20244/40479;
uniform:2345/2622/5229(2400/360/90);plan_us 中位:0.64/3.0/12.7/109ms
(n=32/64/128/256)。其余见 `experiments/mosaic_x1/*/results.csv`。

¹ 2400 = desc 序(新默认)3418.8;720 点 plan 序更优(5068 < desc 5202),
复现需 `-fast_stage_order plan`;其余 6 点两序逐位相同。忠实运行时同步
(`-fast_sync credit`)另加 2.3~3.6%(2400:+11.5%),对比表仍引 async
(对手最有利),见 `experiments/mosaic_x1/33_fast_sync/`。
