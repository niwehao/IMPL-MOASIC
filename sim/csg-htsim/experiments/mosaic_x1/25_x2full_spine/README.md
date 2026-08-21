# 25_x2full_spine — X2-full:per-spine 显式钉法 vs 隐式落点法(§8.2 收尾之二)

pod 布线下 MOSAIC 的 RX-lane 有两种 spine 绑定法:**隐式**(lane m =
第 m 个落点,spine 随 (s+host(d)·G+j)%S 自然轮转)与**显式**
(`-mosaic_explicit_spine`:lane m ≡ spine m%S,落点在该 spine 的合法
集合内轮转)。假设:G%S==0 时隐式轮转天然均匀覆盖;S 与 G 互质时
隐式覆盖有偏,显式应更好。flex_pod,nv=360,leaf 上行 720(2:1),
zipf1.0/seed42,C=S+1。数据 `results.csv`。

## 主表(completion_us)

| S | ECMP | FAST | MOSAIC 隐式 | MOSAIC 显式 | 显式收益 |
|---|---|---|---|---|---|
| 3(互质) | 18243 | 10126 | 8465 | **8393** | +0.9% |
| 4(整除) | 18250 | 14629* | **8348** | 8904 | **−6.7%** |
| 5(互质) | 18254 | 10126 | 8720 | **8487** | +2.7% |

*S=4 的 FAST 值来自 22 组同配置(taper 720)。**FAST 的 S 依赖本身是个
发现**:其 i→i 结构的 spine 指纹 = (8(m+n)+2i) mod S ≡ 2i mod S——
**S 为偶数时恒为偶数,只用得到一半 spine**(S=4:14629,spine 束缚),
S 为奇数时 2i 覆盖全部余数(S=3/5:10126,回到 nv 界)——与 23 组
双平面的奇偶盲区同根同源:同 index 中继 + 模运算钉死 = 系统性类缺失。

## 结论

1. **假设得到证实,且给出精确的使用规则**:`G % S != 0`(互质类)时
   显式钉法更好(S=5 处 +2.7%,隐式的落点→spine 映射有偏);
   `G % S == 0` 时隐式已完美覆盖,显式反而 −6.7%(把落点限制在
   G/S 个候选里,牺牲了 NIC 侧的选择自由度)。
2. 因此 pod 模式的推荐 = **按整除性自动选择**(`G%S==0` 用隐式,否则
   显式)——一行判断,零运行时代价;§8.2 的"S 与 G 互质时覆盖不全"
   疑虑就此闭环。
3. MOSAIC 领先 FAST:S=3/5 处 1.19–1.21×(FAST 未被 spine 束缚),
   S=4 处 1.75×(FAST 只用一半 spine,见上注);领先 ECMP 2.05–2.18×。
   spine 数不改变 pod 结论(22 组)。

复现:22 组命令 + `-flex_spines {3|4|5} -mosaic_corridors S+1
[-mosaic_explicit_spine]`。
