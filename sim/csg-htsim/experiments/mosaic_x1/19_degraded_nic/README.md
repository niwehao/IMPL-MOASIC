# 19_degraded_nic — 未通告的 NIC 降速干扰(问题 8:scale-out 不确定性)

`-flex_degrade_gpu G -flex_degrade_factor F`:GPU G 的 NIC 双向降为
nic/F(建模邻租户干扰/链路降级)。**没有任何调度器被告知**:FAST 的
plan、MOSAIC 的 prior 都按额定 180G 算;执行侧 pacing 按真实速率
(host 知道自己网卡的发送速率,这是硬件事实,对所有方案一致)。
负载 hot-dst zipf1.0/seed42;G=26 是最热 dst(最坏情形),G=7 是冷 GPU
(一般情形)。无降速基线:FAST 3541/10126,MOSAIC 3557/8566(01 组)。
数据 `results.csv`。

## 主表(completion_us)

| nv | 降速点 | F | ECMP | FAST | MOSAIC | M vs F |
|---|---|---|---|---|---|---|
| 2400 | 热 dst 26 | 2 | 35066 | 6864 | **4669** | **1.47×** |
| 2400 | 热 dst 26 | 4 | 70070 | 13301 | **9253** | **1.44×** |
| 2400 | 冷 7 | 2 | 21829 | 4703 | **4540** | 1.04× |
| 2400 | 冷 7 | 4 | 21829 | 9356 | **9007** | 1.04× |
| 360 | 热 dst 26 | 2 | 35066 | **10126** | 10194 | 0.99× |
| 360 | 热 dst 26 | 4 | 70070 | **13301** | 13759 | 0.97× |
| 360 | 冷 7 | 2 | 21829 | 10126 | **8567** | 1.18× |
| 360 | 冷 7 | 4 | 21829 | 10126 | **8335** | 1.21× |

## 结论

1. **强 scale-up(2400)+ 热点降速:MOSAIC 从平手变成赢 1.44–1.47×**。
   基线 regime(01 组)里 2400 是 FAST 主场(1.00×);一块热 NIC 被降速
   后,MOSAIC 的 feedback 把该 dst 的份额推向 RX/SPREAD(NV 入口),
   FAST 的静态 plan 继续按额定速率喂它。这是"执行期不确定性"最直接的
   一组证据:**crossover 不只由拓扑参数决定,也由运行时环境决定**。
2. 冷 GPU 降速:两家都能忍(它不是瓶颈);360 点 MOSAIC 反而略好于
   无降速(8335 vs 8566)——feedback 把该 NIC 的份额挪走,顺带减轻了
   fabric 争用。ECMP 完全不动(21829):它根本没用过那块 NIC 的富余。
3. 如实:360+热点降速时 FAST 略胜(−1~3%)。机理:FAST 的 ingress 本来
   就经 8 块 NIC 摊开(i→i + dispatch),对"单块 NIC"降速有结构免疫;
   MOSAIC 在 nv=360 时 NV 侧替代容量有限,exploration 份额继续撞降速
   NIC。两家都离割界远(13.3k~13.8k vs 界 10069)——单 NIC 降速的割界
   本身偏松(兄弟 NIC 未降速,B2 不变)。

复现:01 组命令 + `-flex_degrade_gpu {7|26} -flex_degrade_factor {2|4}`;
下界:`tools/x1_bounds.py --degrade-gpu G --degrade-factor F`。

## 2026-07-28 交叉引用(32 组)

- 结论 2 的"冷卡降速反而略好"(8335 < 8566)**是 C4 特异的**:32 组在
  当前 SOTA 配置下复核,C6 为 +4.2%(8768)、C6+R 为 +10.7%(8820),
  效应消失并反号;方向分解(新仪器 `-flex_degrade_dir`)显示单独降收/
  降发均不复现(±0.2%),须收发同降。其机制学替代品是 **F6 冷源限速**
  (`-mosaic_cold_pace`,受控 admission,带三道生成期门),见
  `32_paradox_pacing/` 与 `Note/explore/report_paradox_pacing.md` §3。
- 结论 3 的"单 NIC 降速下 FAST 结构免疫、MOSAIC 撞降速 NIC"已由 32 组
  推广成**状态粒度错配**定律(λ 是 per-dst,故障是 per-src-host),并由
  **F7 `-mosaic_host_lambda`** 修复:单中继降速 +32% → +0.6%。
