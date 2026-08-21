# 40_wave_ocs:MOSAIC-O-W——EPS wave/λ 方案照搬到 OCS fabric

**日期**:2026-08-05。**目的**:回答"OCS 方案能否是 EPS 方案的**变体**
而非近似正交的另一套算法"——把 EPS MOSAIC 的全部机制(corridor + λ +
波配额 + 75% 阈值 + 反馈律)原样移植到光电路 fabric,**重配置由 wave
波次触发**,不用 receding-horizon 规划器。实现 = `htsim_optical -wave`
(`src/main_optical.cpp: run_mosaic_ow`),约 700 行,零规划(无候选
枚举、无 scratch 仿真、无 Hungarian)。

## 机制映射(EPS → OCS,详见 `Note/explore/report_wave_ocs_x40.md`)

| EPS | OCS-W |
|---|---|
| corridor(剪枝路径类,4-6 条) | 电路结构类 3 条:D(直连)/F(正环带)/B(反环带) |
| per-dst λ + waterfilling prior | 同;prior = demand+R 闭式(D 列准入、B 带门、环序自动门) |
| wave 配额 `ceil(rem/波余)`、75% 阈值、γ/floor/trust 反馈律 | 逐字照搬,**按 (dst,class) 索引**(串行端口的忠实投影) |
| 固定 fabric,corridor 切分零成本 | **wave 触发重配置**:session 只在波计量首次路由字节时开(付端口 R);未变 corridor 跨波免 R 合并 |
| anti-windup(stale 扣减) | **被 per-class 游标结构性吸收**(卡住的类不推进游标即不会被复喂) |

正典常数不变:W=6、γ=0.7、floor=0.02、threshold=0.75;OCS 新增唯一门
系数 `-wave_gate 2.0`(R 是 EPS 没有的自由度)。全场统一,无逐点调参。

## 主表(n=32,k=1,800G/500ns,4MB chunk,pay-initial;全部 PASSED,cut 核对全过)

completion_us;倍数 = 对手 ÷ O-W(>1 = O-W 胜)。对手为冻结归档:
barrier best(35 组双 oracle 逐点最优)、PR best(39 组每 cell 35 次
async 精确仿真选择的 argmin)。

| demand | R(µs) | **MOSAIC-O-W** | barrier best | ×barrier | PR best | ×PR |
|---|---|---|---|---|---|---|
| uniform | 10 | 2234.2 | 975.6 | 0.44 | 960.6 | 0.43 |
| uniform | 100 | 5653.1 | 2695.7 | 0.48 | ~2674 | 0.47 |
| uniform | 1000 | **7453.1** | 7496.7 | **1.006** | 7482 | **1.004** |
| uniform | 10000 | **20522.2** | 20649.9 | **1.006** | ~20635 | **1.006** |
| zipf1.0 | 10 | 9213.4 | 5423.2 | 0.59 | ~5407 | 0.59 |
| zipf1.0 | 100 | 8475.5 | 7895.3 | 0.93 | 6620.1 | 0.78 |
| zipf1.0 | 1000 | **10275.5** | 13272.3 | **1.29** | 10220.1 | 0.995 |
| zipf1.0 | 10000 | 22676.1 | 22272.3 | 0.98 | 22272.3 | 0.98 |
| zipf2.0 | 10 | 20105.5 | 12983.5 | 0.65 | ~12691 | 0.63 |
| zipf2.0 | 100 | **14182.7** | 15085.9 | **1.06** | 12958.5 | 0.91 |
| zipf2.0 | 1000 | **15982.7** | 15985.9 | **1.000** | 14777.3 | 0.92 |
| zipf2.0 | 10000 | 26886.0 | 24985.9 | 0.93 | 24985.9 | 0.93 |
| persrc | 10 | 3698.0 | 1583.7 | 0.43 | 1583.7 | 0.43 |
| persrc | 100 | 5728.7 | 4158.8 | 0.73 | 4131.9 | 0.72 |
| persrc | 1000 | **7528.7** | 10004.1 | **1.33** | 8253.4 | **1.10** |
| persrc | 10000 | 22205.2 | 19004.1 | 0.86 | 19004.1 | 0.86 |

**读数**:x39 窗口(中大 R × skew)被 EPS 变体以**零规划在线**复刻:
zipf1.0@1000 与 PR 持平(0.995),persrc@1000 **反超 PR 1.10×**、超
barrier oracle 组合 1.33×;zipf2.0@1000 拿到 PR 的 92%。大 R B-gate
自动退化单环(uniform@10000 1.006× 微超归档)。弱点两类,机理明确:
小 R 的 **W 税**(逐波 session roll,W=1 消融即回到 RoN 形态:zipf@10
= 5604.6,距 oracle 3.3%)与中 R 的 **shift 结构缺失**(C=3 无中距
stride 快线;uniform@100 0.48×)。

## 消融与附加实验(全表 results.csv)

- **W 消融**(1/3/6/12):环 regime 平坦(zipf@1000:10283/10381/
  10276/10259——belt 跨波免 R 合并,**波在环侧免费**);D regime 单调
  税(unif@10:1318/1524/2234/3574)。W=6 正典 = 用小 R 的 R 税换
  漂移反应性。
- **反馈消融**(`-wave_static`):persrc@10 反馈 **+21%**(3698.0 vs
  4468.9);其余 D-regime 三点 static 略优(unif@10 −1.4%、zipf@10
  −0.7%、zipf2@10 −0.4%,探索保险费);环 regime 12 点逐位相同
  (方向轴不可测,冻结先验即最优,见 explore 报告 §3/§5)。
- **环序自动门**:persrc 判 NN(@100/@1000 −11~−14%),zipf 判恒等
  (强制 NN 在 zipf2@1000 +10.5% 劣化,门方向正确)。
- **dwindow**:0(严格短向)全域 ≥ dw4,已设默认;dw4 消融行保留。
- **运行中漂移**(Δ=0.5×seed44 @ t,R=1000;对照 39 组归档):

| 注入 t(µs) | O-W(纯配额律吸收) | pr_online(39) | 串行 async | vs 串行 | vs pr_online |
|---|---|---|---|---|---|
| 3000 | 13043.5 | 12922.6 | 16717.0 | 1.28× | 0.99× |
| 5000 | 14470.4 | 14644.7 | 16717.0 | 1.16× | 1.01× |
| 8000 | **14522.3** | 19695.2(crossover) | 16717.0 | 1.15× | **1.36×** |
| 9500 | 15018.0 | 16913.3 | 16717.0 | 1.11× | 1.13× |

  **pr_online 的晚期漂移 crossover 被治愈**(era 世代 key 重建 +
  波配额自然吸收):四个注入点全部 13.0-15.0ms,x39 §7(f) 的
  "吸收/串行自动门"开放项消解(无需门)。R=100@3000:11196.6
  (vs 串行 1.17×)。
- **n=64**:zipf@1000 = 22635.3(vs barrier **1.92×**,vs PR 0.91);
  persrc@1000 = 32244.3(vs barrier 1.11,vs PR 0.62);zipf@100 =
  20835.3(vs barrier 0.68)——n64 中 R 由 shift 结构主导(39 组
  ron_async 中等 d),C=3 缺该族,如实报告。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_optical
# 单 cell
./sim/tests/htsim_optical -wave -nodes 32 -reconf_us 1000 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# 漂移
./sim/tests/htsim_optical -wave -nodes 32 -reconf_us 1000 \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt \
  -drift_file cache/optical/drift_half_seed44.txt -drift_us 8000
# 全批(log = cache/optical/x40_wave.log,DONE 标记,md5 头)
bash script/optical_wave.sh
```

开关登记见 `Note/var.md`(40 组段);机制细节、失败迭代全记录与
审稿自查见 `Note/explore/report_wave_ocs_x40.md`。
