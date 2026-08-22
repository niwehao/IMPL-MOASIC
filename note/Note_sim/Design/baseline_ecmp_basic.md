# Baseline:static_ecmp 与 basic(无 schedule 的两个下限参照)

> 2026-07-12。两者都不构造 schedule,是"什么都不调度"的参照系。
> **对比实验的铁 baseline 是 static_ecmp**(CLAUDE.md 铁律)。

## static_ecmp

- **构造**:启动时每 (src,dst) 用确定性哈希在等价最短下一跳里选**一条**
  固定路径(`scheduler/static_ecmp.h:13` `install_paths`,`build_path`
  :167 的线性同余 salt)。同一 pair 永远同路,不逐包散列、不感知负载。
- **执行**:workflow runtime 一次性全量入池,pool-order 选包。
- **特性**:
  - 对 nv 不敏感(跨机全走 NIC 直连;热点单 NIC 是墙)——所以 ECMP 行
    在 nv 扫描里是常数(21829@zipf1.0),各 α 也各是常数(12675/32998/
    43421);
  - 对 fabric 钉死路径的退化最敏感:taper(brick 23533@60G)、pod hash
    撞车(20347@4:1)、双平面 B 面降速(**线性崩溃**:25040/50264/
    100710)——它是"路径不可变"代价的量尺;
  - 偶尔意外地好:机内独大形态 2h16g@90(21572,比 MOSAIC 还好)、
    弱 NV 直连图——"什么都不做"避开了所有中继开销(报告 §7.1 M6)。

## basic

- 逐跳动态选下一跳(forwarding_choices + 三类 split 权重,
  `scheduler/basic.h:100`),等价"包级动态喷洒 + cut-through"。
- 在全对全直连的 flex/dgx 系列上退化 ≈ ECMP(报告 §5 如实注明);
  在直连稀疏图(regular_32)上是**全场最快**(3638,比一切 solver 好)
  ——那是它的主场,也是 MOSAIC 的 M2 不适用区(报告 §7.1)。

## 用法

```bash
# 两者都不需要 schedule 文件/求解:
./sim/tests/htsim_smart_regular ... -scheduler static_ecmp   # 或 basic
# 非均匀直接吃 -demand_file(与 fast/mosaic 相同路径)。
```

## 报告里的角色

- 所有对比表的 1× 基准列;
- ECMP 的结构下界(如 uniform+taper 的每 leaf 168MB/taper)用来验证
  它贴的是哪堵墙(14 组 README 有推导);
- 审稿意义:任何方案若输给 static_ecmp(如 FAST@nv90、pod 4:1、
  TE-CCL@90),即证明其结构假设在该 regime 反噬。
