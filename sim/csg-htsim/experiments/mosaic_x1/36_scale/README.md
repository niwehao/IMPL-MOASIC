# 36 组:超大规模 weak scaling(n=256/512/1024)

**日期**:2026-08-01。**契约**:per-src 总量恒定 65MB(base =
2MB·31/(n−1)),hot-dst zipf α=1.0 seed42,padded 16MB;拓扑 =
flex_rail hosts×8 nv360/nic180/taper180。
**政策**:单次模拟 >20min 自动放弃(SKIPPED);LP 系 solver 排除
(声明:约束求解在数百节点无解,TACCL 32 节点已 11-17min)。

- `results.csv`:整理表(FAILED 仅诊断、SKIPPED 注明原因);
  `eps_raw.log` = 原始批日志。
- 关键结论见 `Note/explore/2026-0801-x36-scale-1024.md`:MOSAIC+PC 贴界
  (1.024×/1.065×)、FAST 需 4-16GB buffer 且 **per-A2A 构造
  111ms→1.49s→2.70s(比 MOSAIC 85-343×,256 起构造>collective
  本身)**。
- **仿真器优化(本组做的,全库逐位回归)**:chunk pool vector→list
  (O(1) requeue)+ per-epoch FIFO 桶索引(min-epoch O(log));
  n=128 提速 2.46× 且 completion 逐位不变;n=32 七值回归电池全过
  (8565.57/8418.42/3525.76/7939.84/10126.4/3418.8/21829.2)。

复现:`script/x36_eps_scale.sh` / `x36_fast_buffer.sh`。
