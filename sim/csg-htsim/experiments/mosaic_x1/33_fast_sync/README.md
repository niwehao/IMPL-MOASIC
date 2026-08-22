# 33_fast_sync — FAST 运行时 stage 同步(none/credit/barrier)与 stage 顺序修正

2026-07-28。主报告:`Note/explore/2026-0728-x33-fast-runtime-sync.md`(动机、官方源码
逐行映射、实现、审计全文)。

## 设置

- flex_rail n32(4×8,nic 180,taper 180),store_forward,packet 1KB;
- zipf:`demand_n32_zipf1.0_seed42.txt`,chunk 33554432;
  uniform:message=chunk=2097152;
- FAST 统一带 `-fast_alpha_beta_guard`(公平性义务);
- 变量:`-fast_sync {none,credit,barrier}` × `-fast_stage_order {plan,desc}`;
- 全部 validation PASSED、queue_drops 0、ecn_rate_cuts 0。

## 旋钮语义(一句话)

- `none`(默认,历史行为):chunk availability 是唯一硬约束,stage 全异步
  ——**缺失官方运行时同步**,是对 FAST 有利的上界近似。
- `credit`:忠实移植官方 NVSHMEM kernel 执行期门控(rank balance 收齐→
  首 inter;per-rank stage 串行;接收端双缓冲 credit(收齐 p-1 且
  dispatch 完 p-2 才授信 p);dispatch 等本 stage 收齐)。
- `barrier`:官方 simulation 分析模型的全局 stage 屏障(同步代价上界)。
- `desc`(新默认):官方 production 的 largest-freq-first stage 序;
  `plan` = 旧行为(Birkhoff 提取序),复现 ≤32 组归档数字用。

## 主表(completion_us;credit/barrier 均为 desc 序)

| workload | nv | none(plan) | none(desc) | credit | barrier | credit 开销 | barrier 开销 |
|---|---|---|---|---|---|---|---|
| zipf | 2400 | 3541.18 | **3418.80** | 3812.28 | 3862.40 | +11.5% | +13.0% |
| zipf | 720 | **5067.69**¹ | 5201.67 | 5321.09 | 5816.92 | +2.3% | +11.8% |
| zipf | 600 | 5988.27 | 5988.27 | 6203.85 | 6811.11 | +3.6% | +13.7% |
| zipf | 480 | 7369.14 | 7369.14 | 7600.27 | 8314.54 | +3.1% | +12.8% |
| zipf | 420 | 8746.16 | 8746.16 | 8984.31 | 9843.75 | +2.7% | +12.6% |
| zipf | 360 | 10126.4 | 10126.4 | 10384.1 | 11384.9 | +2.5% | +12.4% |
| zipf | 180 | 20243.8 | 20243.8 | 20728.9 | 22670.8 | +2.4% | +12.0% |
| zipf | 90 | 40478.6 | 40478.6 | 41430.7 | 45242.7 | +2.4% | +11.8% |
| uniform | 2400 | 2344.85 | 2344.85 | 2445.04 | 2445.04 | +4.3% | +4.3% |
| uniform | 360 | 2622.38 | 2622.38 | 2722.46 | 2722.46 | +3.8% | +3.8% |
| uniform | 90 | 5229.20 | 5229.20 | 5257.37 | 5258.09 | +0.5% | +0.5% |

¹ zipf@720 plan 序未在本批重跑,引自 21 组归档(代码 none+plan 路径与
历史逐位一致,本批 2400/480/360/90 四点已验证)。开销百分比相对
none(desc)。zipf 9 stages,uniform 3 stages(等 freq,排序无效果)。

## stage 顺序修正对同步模式的作用(nv360)

| sync | plan 序 | desc 序 | 修正收益 |
|---|---|---|---|
| none | 10126.4 | 10126.4 | 0 |
| credit | 10658.1 | 10384.1 | −2.6% |
| barrier | 12401.6 | 11384.9 | −8.2% |

不做 desc 修正会系统性高估 FAST 的同步开销(对 FAST 不公平)。

## 结论

1. **async(none)确实高估了 FAST**,但幅度有限:忠实 credit 语义下
   zipf 中低 NV 只慢 2.3–3.6%,不改变任何既有对比结论的方向;历史
   FAST 数字作为"对手最强形态"引用仍然合法(方向保守)。
2. **例外是强 NV(zipf@2400):credit +11.5%**——NVLink 越快,async
   掩盖的 stage 串行气泡越大。同时 desc 序让 async FAST 在该点快 3.5%
   (3541→3419):**SOTA 2400 行 vs FAST 从 1.04× 收窄到 1.004×**
   (MOSAIC C6+R 3405 vs 3418.8),已如实更新 SOTA.md。
3. uniform(FAST 主场)同步开销可忽略(0.5–4.3%),credit≈barrier
   (3 个等 freq permutation,pairwise 门与全局屏障几乎等价)——
   FAST 论文"平衡负载无 straggler"的自洽性在包级复现下成立。
4. stage order 对 async 影响不单调(2400 −3.5%、720 +2.6%、其余 0):
   引用 FAST async 基线时按点取两序较优者(对手最有利原则),
   即 2400 用 desc 3418.8,720 用 plan 5067.69,其余不变。

## 复现

```bash
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
  -packet_size 1024 -chunk_size 33554432 -scheduler fast_schedule \
  -fast_alpha_beta_guard -fast_sync credit -workload A2A -all_to_all \
  -store_forward -quiet -demand_file $D
# -fast_sync {none|credit|barrier} × -fast_stage_order {desc|plan}
# uniform:去 -demand_file,-chunk_size 2097152 -message_size 2097152
```

批日志:`x33_fast_sync_planorder.log`(plan 序)、`x33_fast_sync_desc.log`
+ `x33_fast_sync_desc2.log`(desc 序)。
