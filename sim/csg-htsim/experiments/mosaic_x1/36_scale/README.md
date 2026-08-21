# 36 组:超大规模 weak scaling(n=256/512/1024,EPS+OCS)

**日期**:2026-08-01。**契约**:per-src 总量恒定 65MB(base =
2MB·31/(n−1)),hot-dst zipf α=1.0 seed42,padded 16MB;EPS =
flex_rail hosts×8 nv360/nic180/taper180;OCS = k1 800G 16MB chunk。
**政策**:单次模拟 >20min 自动放弃(SKIPPED);LP 系 solver 排除
(声明:约束求解在数百节点无解,TACCL 32 节点已 11-17min)。

- `results.csv`:整理表(FAILED 仅诊断、SKIPPED 注明原因;OCS 的
  ron d≤4 行科学计数截断损坏已排除——模型值差 ≥87×,不影响赢家);
  `eps_raw.log`/`ocs_raw.log` = 原始批日志。
- 关键结论见 `Note/explore/report_scale_1024.md`:MOSAIC+PC 贴界
  (1.024×/1.065×)、FAST 需 4-16GB buffer 且 **per-A2A 构造
  111ms→1.49s→2.70s(比 MOSAIC 85-343×,256 起构造>collective
  本身)**、OCS R=1ms 处 MOSAIC-O 静态族胜 RoN 网格最优 1.42-1.46×。
- **仿真器优化(本组做的,全库逐位回归)**:chunk pool vector→list
  (O(1) requeue)+ per-epoch FIFO 桶索引(min-epoch O(log));
  n=128 提速 2.46× 且 completion 逐位不变;n=32 七值回归电池全过
  (8565.57/8418.42/3525.76/7939.84/10126.4/3418.8/21829.2)。
  OCS scale-mode(n>64):König 闭式 + 贪心匹配 + 解析剪枝 top-3
  精确评估 + BVN streak(64 相位)。

复现:`script/x36_eps_scale.sh` / `x36_fast_buffer.sh` /
`x36_ocs_scale.sh`(n256+n512R10)+ `x36_ocs_resume.sh`(其余)。
