# 35 组:可重配置光网络三方对比(RoN / Harvest / MOSAIC-O)

**日期**:2026-07-31。**设定**:n=32、k=1(每节点 1 收发器)、800Gbps、
500ns/hop、4MB chunk、store-and-forward、phase barrier、pay-initial R
记账(与 Reconfigure-or-Not Thm.1 一致);R ∈ {1,10,100,1000,10000}µs
覆盖可调激光(ns-µs)→ 电光(µs)→ MEMS(ms)→ patch panel(10ms+)。
仿真器:`src/optical_fabric.h` + `src/main_optical.cpp`(chunk 级,
与 EPS 分离;selftest 逐位复现 RoN 动机实例)。

**方案**:
- `ron_auto`:Reconfigure-or-Not,自身 model 选 d(deployable);
- `ron_best`:RoN oracle——d=1..31 全部仿真取最优(对对手最有利,
  任何在线系统不可得);
- `harvest`:Harvest θ-最优静态拓扑(Gurobi MILP,k=1;平稳 demand
  序列下其区间 DP 退化为单拓扑,数学论证见报告 §4);
- `mosaic_o`:在线 receding-horizon(Birkhoff 匹配计划 ∨ Cayley shift
  计划、残差反馈、15% 滞回;全表同一套参数,无 per-R/per-demand 调参)。

**demand**:uniform 2MB/pair、hot-dst zipf α=1.0/2.0(seed42)、
persrc α=1.0;漂移组 prior=seed44→actual=seed42。
**下界**:max-line cut(k=1 单收发器,max(最大行和, 最大列和)·8/BW)。

结果:`results.csv` + `plot.png`;mosaic_o 列 = v12 定稿(三族候选:
Birkhoff ∪ 双策略 Cayley 截断族 ∪ 静态环族,精确仿真评估;二进制
md5 `fda6485391efb7e390468a152202a68e`);zipf20 三个大 R cell 使用
离线 θ-opt 环序候选(`cache/optical/theta_order_*.txt`,从 Gurobi 解
抽取,LP-表先例);`zipf10_runtime_drift` 行 = 运行中漂移实验
(`-drift_file/-drift_phase`,含 RoN/Harvest 串行对照);
`*_k2` 行 = k=2 扩展(`-degree 2`:双匹配剥离/±s 双向 Cayley/双向环
+ `-static_topo_file` θ-artifact;RoN 依公平原则不入 k2 表;二进制
md5 `27dca612c1bfb8e381d9cde4f5e18e70`,k=1 行为逐位不变已回归)。
结论与读数见 `Note/report_mosaic_ocs.md` §5。

复现:`script/optical_compare_v2.sh`(约 20-40 分钟,内含 Gurobi
θ-opt;`FORCE` 无缓存,直接重跑)。
