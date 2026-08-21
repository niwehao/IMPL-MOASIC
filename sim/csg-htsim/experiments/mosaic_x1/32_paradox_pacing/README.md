# 32_paradox_pacing — 带宽/降速双悖论、状态粒度、挂账结案

2026-07-28。回答三个问题 + 一条挂账(完整叙事:
`Note/explore/report_paradox_pacing.md`;常设配置表:`Note/SOTA.md`)。
全部实验在**冻结二进制**上执行(`cache/nonuniform/x32bin/`,批日志头
记录 md5):诊断用 `htsim_f1f5b2a`(= 提交 f1f5b2a 原样),机制验证用
`htsim_mech`(f1f5b2a + F6/F7/门,flag-off 逐位回归通过)。

## 数据

`results.csv` 由 `consolidate.py` 从六份批日志汇总(section 列):

| section | 日志 | 内容 |
|---|---|---|
| acct | x32_acct.log | 挂账:23 组 rail2p 九点全网格重跑 + 24 组 C5/known 复核 |
| p1rail | x32_p1rail.log | 问题1:taper × {base,static,R} × 5 demand + solo 对照(trace 全日志另存 x32_trace_t{90,180,360}.log) |
| p1pod | x32_p1pod.log | 问题1:pod S×C 解耦网格(总上行固定/每 spine 固定,隐式/显式) |
| p23 | x32_p23.log | 问题2/3:双向降速矩阵(g7/g2/g24/g26 × 2/4 × C4/C6/R/FAST)+ 扩 C 证伪 |
| p23b | x32_p23b.log | 问题2/3:方向分解(-flex_degrade_dir)、中继集合(1/3→3/3)、oracle 冷入口限速 |
| mech | x32_mech.log | F6/F7 全面验证:rail 8 nv、门(uniform/persrc/taper)、稳健性、其他拓扑、F7 降速矩阵 |

## 结论速览

1. **挂账**:23 组 MOSAIC 归档产生于 X4 原地替换掉的旧 corridor 族,
   任何已提交版本不可复现(v5 与当前逐位一致;ECMP/FAST 逐位同旧归档);
   重跑后**九点全 ≥ FAST(1.02×–2.22×),0.73× 作废**;24 组自身数字
   逐位可复现。23/24 README 与 `report_mosaic_eps.md` 已改写。
2. **问题1**:solo 只跑热点时 taper 单调受益(7936→7519,贴界 7552);
   全量 workload 下热点 +463µs 而 19/32 dst 变快(同轨冷 dst
   −1.1~−2.1ms)——**变长的是热点自己的尾巴**,p99 变好是分位数 vs
   最大值。"5%"是抖动不是定律(seed43 下 t90 反而最差)。spine 3→5
   变慢 = 25 组 C=S+1 混淆;解耦后同 C 无单调性。真病灶 =
   explicit-spine 在 C−1<S 时塌带宽(S6/C2 18162 vs 隐式 9821),
   已加生成期可达性门。
3. **问题2**:19 组"降速反而快"只在 C4 成立(C6 +4.2%、R +10.7%);
   方向分解:单独降收/降发不复现(±0.2%)。限速 ≠ 优先级(FIFO 不可
   抢占 + work-conserving);oracle 冷入口 ×2 限速仍 −4.0% ⇒ 做成
   **F6 `-mosaic_cold_pace`**(三道生成期门:skew / 2× slack /
   **taper<nic 有效性门**)。rail 8 点全 ≤ base、7 点刷新 SOTA
   (15567/11727/7917/7180/6370/5480/4885/3405,距界 +2.3~+7.8%);
   uniform/persrc/窄 taper 门关逐位=base。
4. **问题3**:λ 粒度 (dst×corridor) ≠ 故障粒度 (src-host×corridor)。
   单中继降速 MOSAIC +32%/FAST 免疫(0.91×);全类中继降速 1.31×
   (类内对称,λ 可表达);降热点自身 1.00×(物理)。**扩 C 证伪**
   (C3–C6 全 11.1–11.6k;C2=9242 反而最好,因整类绕开故障角色)。
   **F7 `-mosaic_host_lambda`** 把单中继场景修到 8471(+0.6%,回到
   1.20× vs FAST);干净场景 8403 中性;E2 再锚定用乘法同步保留已学
   偏差。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
bash cache/nonuniform/x32_acct.sh      # 挂账网格
bash cache/nonuniform/x32_p1rail.sh    # 问题1 rail(含 trace)
bash cache/nonuniform/x32_p1pod.sh     # 问题1 pod S×C
bash cache/nonuniform/x32_p23.sh       # 问题2/3(双向降速)
bash cache/nonuniform/x32_p23b.sh      # 问题2/3(方向/集合/oracle)
bash cache/nonuniform/x32_mech.sh      # F6/F7 验证
python3 experiments/mosaic_x1/32_paradox_pacing/consolidate.py
```
