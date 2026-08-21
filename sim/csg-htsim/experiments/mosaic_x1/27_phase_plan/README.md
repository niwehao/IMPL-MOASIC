# 27_phase_plan — E4:生成期流体相位计划(份额模型被否,序预测保留)

检验"demand 全局已知 ⇒ 分段静态计划可替代运行时 feedback"。
`-mosaic_phase_plan`:构造期 progressive-filling 流体模拟(资源 =
每 GPU NIC-ingress lane + 每 dst NV-ingress 管;子流 max-min 公平),
预测完成序 + 每相各 dst 的 nv_share 表;运行时第 k 个 dst 完成即切到
第 k 相 λ。生成开销 plan_us=177ms@n32(对比 MOSAIC 常规 20–195µs)。

## 结果(zipf1.0/seed42,C=6,µs)

| nv | bound | base | plan+feedback | plan+static |
|---|---|---|---|---|
| 90 | 15104 | 16276.9 | 18016 | 23743 |
| 180 | 11328 | 12672.7 | 13584 | 16136 |
| 360 | 7552 | 8418.4 | 8379 | 9048 |
| 2400 | 3327 | 3525.8 | 3775 | 3563 |

**份额模型被否**:max-min 按子流计数公平(每 cross pair 1 direct +
G−1 relay 子流)结构性偏向 NV 侧,给出 nv_share≈0.48,而 nv90 最优
= 0.33 ⇒ NIC 管欠载。plan+static 崩、plan+feedback 靠 feedback 拉回
但仍多付学习代价。

**序预测保留**:nv90 预测 last-3 (10,5,26) 与实测完全一致;nv360
Spearman 0.804、last-2 (5,26) 全中。⇒ demand 已知时完成序高度可预测,
这是 26 组 E3(skew 优先序)offset 的依据。

**理论归结**(见报告 §5):makespan 最优流体计划 = hot 从 t=0 以
bv+bn 满速(=B1),cold 填缝;该计划的 λ 就是 isolated prior(时不变)。
⇒ "计划"的正确载体是**优先序**(26 组 E3,静态、零开销),不是 λ 相位表;
相位表想赢就必须建对 lane 级争用模型,而错误的争用模型比不用更糟。

数据 `results.csv`;日志 `cache/nonuniform/x26_matrix.log`(plan/planst 行);
序预测对照 `cache/nonuniform/x27_e4_nv360_order.txt`。
报告:`Note/explore/report_donation_phase.md`。
