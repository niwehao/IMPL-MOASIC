# 18_synthesis_speed — schedule 合成速度 vs 规模(问题 6、11)

同一 collective(hot-dst zipf1.0/seed42 或 uniform 2MB)的 schedule 生成耗时。
口径:MOSAIC = prior 构造(ctor + waterfilling,mosaic_plan_us 打点);
FAST = 完整 plan(balance + Birkhoff 分解 + dispatch,fast_plan_us);
solver = 求解(+转换),n=32 复用先前实测,n=64 本轮实测(timeout 30min)。
在线方法取 3 次中位数。数据 `results.csv`,图 log-log。

## 主表

| 方案 | n=32 | n=64 | n=128 | n=256 |
|---|---|---|---|---|
| **MOSAIC** | **20µs** | **72µs** | **195µs** | **816µs** |
| FAST | 636µs | 3.0ms | 12.5ms | **108.8ms** |
| SyCCL | 40.5s | 262s | — | — |
| TE-CCL | 147s | **>1800s 超时** | — | — |
| TACCL | ~840s | 跳过(sketch 需人工重设计;32 已 840s) | — | — |

## 结论

1. **量级鸿沟**:MoE 流量矩阵几百 ms 一变;MOSAIC 每次到达现算 prior
   仅 µs 级(O(N²) 闭式),256 GPU 仍 <1ms——"快速及时合成大 schedule"。
2. **FAST 超线性**(Birkhoff 阶段数 9→49→225→961,匈牙利分解 O(S⁴)量级):
   256 GPU 达 108.8ms,已接近 MoE 迭代间隔的同数量级;MOSAIC/FAST 的
   plan 时间比从 32× 拉大到 133×。
3. **离线 solver 在 64 GPU 就撞墙**:TE-CCL 30 分钟无解(32→64 节点 LP
   变量数爆炸);SyCCL 262s(6.5× on 2×规模);TACCL 的 sketch 本身要人工
   重写。即使解得出,分钟级延迟对动态负载意味着"解出来时矩阵已经换了"。
4. 公平注记:solver 解的是 2MB uniform(它们的最优场景);非均匀还要
   叠 pad-to-max 重解(13/15 组),耗时只增不减。

复现:在线方法 `timeout 240 ./sim/tests/htsim_smart_regular ... | grep -m1
{mosaic,fast}_plan`;solver 见 `cache/nonuniform/x18_solver_n64.log`。
