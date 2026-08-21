# 20_host_shape — host 形态变化:corridor 剪枝的泛化(问题 9)

同一张 32 GPU demand 矩阵(zipf1.0/seed42,pair 结构与 host 划分无关),
改 flex_rail 形态:8host×4gpu(G=4:SPREAD lane 只剩 2 条可选轨)与
2host×16gpu(G=16:机内为主、每 host 16 块 NIC)。nv 语义 = 每 GPU
scale-up 带宽,保持同值可比。数据 `results.csv`。

## 主表(completion_us)

| 形态 | nv | bound | ECMP | FAST | MOSAIC | M vs F |
|---|---|---|---|---|---|---|
| 8h×4g | 2400 | 6668 | 27831 | **7901** | 7942 | 0.99× |
| 8h×4g | 360 | 7552 | 27831 | 9076 | **8976** | 1.01× |
| 8h×4g | 90 | 15104 | 27831 | 34690 | **16113** | **2.15×** |
| 2h×16g | 2400 | 1581 | 14574 | 2341 | **2147** | **1.09×** |
| 2h×16g | 360 | 7552 | 14574 | 10842 | **8487** | **1.28×** |
| 2h×16g | 90 | 21773 | 21572 | 43342 | **23523** | **1.84×** |

## 结论

1. **crossover 结构在两个极端形态都成立**:弱 scale-up 侧 MOSAIC 全胜
   (2.15× / 1.84×),强侧平手或反超。corridor 集合(§4.1 剪枝)对 G=4
   (SPREAD 轨变少)和 G=16(机内为主)都不需要任何改动。
2. **G=16 时 MOSAIC 在 nv=2400 也赢 1.09×**:host 数只有 2,FAST 的
   server 级 Birkhoff 只剩一个非平凡置换,全局协调优势消失;MOSAIC 的
   per-dst 混合不依赖 server 数。
3. 8h×4g@90:FAST(34690)比 ECMP(27831)还慢——W7 在小 G 下更痛
   (balance/dispatch 的 NV 代价摊不开)。MOSAIC 16113 距界 +6.7%。
4. 如实:2h×16g@90 ECMP(21572)略好于 MOSAIC(23523)。该点跨机
   只有 16 对等效直连、机内占主导,ECMP 的"什么都不做"恰好躲开了所有
   中继开销;MOSAIC 的探索费在极端形态下没赚回来(距界 +8.0%)。

复现:01 组命令 + `-flex_hosts {8|2} -flex_gpus_per_host {4|16}`;
下界:`tools/x1_bounds.py --gpus-per-host {4|16}`。
