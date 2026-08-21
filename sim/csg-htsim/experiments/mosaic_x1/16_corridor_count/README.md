# 16_corridor_count — 为什么是 4 条 corridor?(问题:corridor 数消融)

`-mosaic_corridors C` 嵌套集合:C=1 {DIRECT};C=2 +RX;C=3 +TX;
C=4 +SPREAD3(默认);C=5/6 再加额外 SPREAD lane(同 SPREAD3 形状、
每 pair 选不同的过境 rail j——只加 λ 粒度,不加新的资源签名类)。
负载 hot-dst zipf1.0/seed42(另附 zipf1.5@90 与 seed43@360 稳健性点),
flex_rail,32 GPU。数据 `results.csv`(hop_overhead_x = hop 字节 / app 字节)。

## 主表(completion_us,zipf1.0/seed42)

| C | 2400 | 360 | 90 |
|---|---|---|---|
| 1 | 21829(≡ECMP,逐位一致) | 21829 | 21829 |
| 2 | 3674 | 8966 | 17048 |
| 3 | 4067 | **8361** | **16060** |
| 4(默认) | 3557 | 8566 | 16368 |
| 5 | **3470** | 8485 | 16177 |
| 6 | 3526 | 8418 | 16277 |

稳健性点:C=6 vs C=4 —— zipf1.5@90:27608 vs 28198;seed43@360:8527 vs 8633。

## 结论

1. **C=1 严格退化为 ECMP**(sanity:单 DIRECT corridor 没有任何摊开手段)。
2. **1→2(加 RX)是最大的一跳**(21829→3674@2400):接收侧摊开是收益主体。
3. **没有全局最优的 C**:2400 上 C=5 最好,360/90 上 C=3 最好;但相邻档差距
   只有 1–2.5%,而 C=1/2 的缺口是数倍——**结构完备性(每类资源签名至少
   一条)重要,数量本身不重要**,这正是 §4.1 剪枝论证的实验版。
4. **C=6 在全部 5 个测点上一致优于 C=4**(+0.6%~2.1%),C=5 在 2400 上
   已越过 FAST(3470 vs 3541)。设计迭代含义见 21_design_iter。
5. hop 开销单调随 C 增长(2400:1.57×→1.77×),但换来的端到端收益为正
   ——多 lane 把 SPREAD 类的轮转负载均衡做得更细。

复现:`./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail
-flex_nv_gbps {nv} -packet_size 1024 -chunk_size 33554432 -scheduler
mosaic_schedule -workload A2A -all_to_all -store_forward -quiet
-demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt -mosaic_corridors {C}`
