# x45:-mosaic_decision_trace —— 决策轨迹 dump 开关(真机对拍用)

**日期**:2026-08-14。**性质**:纯诊断输出,不改任何决策路径;默认关 =
旧行为逐位不变。

## 开关

```
-mosaic_decision_trace <file>
```

每波每 (dst, src) 写一行:`W <wave> D <dst> S <src> Q <quota> SH <c0..c5>`
(SH = 六条 corridor 的字节切分,机内 pair 记在 c0/DIRECT)。代码位置:
`scheduler/mosaic_schedule.h: set_decision_trace()` 与 `emit_wave_body`
的两处 fprintf;`src/main_regular.cpp` 解析。

## 为什么需要

真机实现(`Impl/mosaic-rt`,Imp_Plan/01 §4)把 L2 协议核逐函数移植到
CPU runtime,移植等价性需要金标对拍:同一 demand 文件驱动模拟器与
runtime 的注水核,逐 (wave,dst,src) 比走廊切分。本开关就是金标的来源。

## 首轮对拍结果(2026-08-14,n=8 2×4 rail,nv400/nic100/taper180,U2b 全开)

wave0(t=0 prior)在 hotdst-42 / hotdst-α1.5-43 / khot4 / persrc 四个 cell
上 4×56 行**逐字节一致**。逼出并修复了 runtime 侧 5 处移植分歧
(SPREAD 轨选择、同轨 lane 数、注水步长、tie-break 基准、fabric 账缺失),
细节见 `Impl/mosaic-rt/log.md` 坑 #18。模拟器侧零行为变化。
