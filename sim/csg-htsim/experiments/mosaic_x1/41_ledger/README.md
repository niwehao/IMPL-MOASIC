# 41_ledger:MOSAIC-L——全局资源账本分配核 vs 原方案(λ 反馈)

**日期**:2026-08-05。**目的**:验证 `Note/explore/report_eps_foundation_audit.md`
的立足点批判——"λ 学的是本可记账的东西"。实现 `-mosaic_ledger`
(`scheduler/mosaic_schedule.h`,默认关,legacy 全逐位):每资源
(NIC↑↓/NV↑↓/(leaf,spine) 段)记 outstanding/有效速率,波配额按
max-投影完成水平化注水;λ/censoring/trust/anti-windup/floor 全不用;
测量只剩 **NIC 域异常检测器**(忙时聚合吞吐 <0.45×额定 ×3 折、
quorum≤2 门)。冻结常数:band 5%、chain 切片 1MB、折算窗 500µs
(在 seed42 主线上定,其余 50+ cell 盲跑 = 事实上的 held-out)。

## 主线(zipf1.0/seed42,C6;两套对照,倍数 = 对手÷本方案)

**核对核**(ledger 单独 vs 21 组裸 C6 λ 反馈)与
**栈对栈**(LU = ledger+PC+PACE vs 38 组 MOSAIC-U SOTA):

| nv | bound | 裸 C6 | **ledger** | ×C6 | U(SOTA) | **LU** | ×U | LU 距界 |
|---|---|---|---|---|---|---|---|---|
| 2400 | 3327 | 3526 | 3465.3 | 1.018 | 3506.2 | **3413.6** | **1.027** | +2.6% |
| 720 | 4532 | 4960 | 4742.6 | 1.046 | 4812 | **4650.5** | **1.035** | +2.6% |
| 600 | 5229 | 5610 | 5353.7 | 1.048 | 5441 | **5350.5** | **1.017** | +2.3% |
| 480 | 6181 | 6590 | 6224.7 | 1.059 | 6296 | **6253.2** | **1.007** | +1.2% |
| 420 | 6798 | 7542 | 7114.8 | 1.060 | 7138.3 | **7077.4** | **1.009** | +4.1% |
| 360 | 7556 | 8418 | 8057.1 | 1.045 | 7939.3 | **7907.2** | **1.004** | +4.7% |
| 180 | 11330 | 12673 | 11607.5 | 1.092 | 11711.1 | 11718.3 | 1.000 | +3.4% |
| 90 | 15105 | 16277 | 16373.0 | 0.994 | 15749.4 | 16571.7 | **0.950** | +9.7% |

- 核对核:7/8 胜(+1.8~9.2%),90 平手(−0.6%);
- 栈对栈:7/8 ≥ SOTA(5 点净胜 +0.7~3.5%,180 逐位平),**2400 vs
  FAST(3418.8,33 组最强口径)从 0.975× 翻正为 1.002×**——SOTA 表
  唯一落后点被修复;90 落后 −5.0%(唯一 regression,如实);
- 距界:480 → **+1.2%**、600 → +2.3%、2400 → +2.6%(U 为 +1.9%/
  +4.1%/+5.4%)——审计 H2 的"中段 gap 主体 = 跨 dst 失衡"成立。

## 各 suite 裁决(全部 65+18 运行 PASSED、drops 0;details in results.csv)

- **H4 非通告降速(旗舰)**:8/8 cell 胜归档。hot×4@2400 =
  **3827.6**(vs C6 8892 = **2.32×**、vs FAST 13301 = 3.48×,距降速界
  仅 **+4.3%**);cold×4@2400 = 3488.4(vs 9007 = **2.58×**,比自身
  无降速 3465 只多 +0.7% = 几乎完全吸收);LU 全栈复验 3966/3507 保持
  2.24×/2.57×。机理:λ 反馈只有 per-(dst,corridor) 粒度,账本按
  **单资源**检测+绕行。
- **H5 探索保险费(M3)**:uniform+taper 四点全胜(tp60@2400:2430 vs
  归档 3164 = 1.22×,距 FAST-免疫值只剩 −3.5%,曾 −26%);
  uniform@360/@90 = 2473/4033,**反超 FAST 1.06×/1.30×**(M1 区间
  部分翻盘,无 PC 亦胜);无 floor ⇒ 不再有强制探索流量。
- **稳健(16 cell)**:13 胜(persrc@90 **1.25×**、zipf1.5@2400
  1.12×、zipf2.0@2400 1.10×),4 败集中在 **@90×hot-dst**(seed43
  0.949/seed44 0.982/zipf1.5 0.951/zipf2.0 0.943)——极端饱和下
  pause 耦合的有效速率仍属测量反馈的残余领地(开放项);
  zipf0.5@2400 0.997 平。
- **漂移**:stale prior 与 fresh **逐位相同**(2532.06)——账本不读
  demand 先验,结构性免疫;且 2532 比归档 adapt 最好值 2726 快 7.7%。
- **taper(非均匀)**:7868/7914 vs 8212/8675(1.04×/1.10×)——
  无反馈的 taper 自适应(spine 段在账本里可见)。
- **H6 pod**:8294(vs 22 组 C5 8348 = 1.006)/6859(vs 6788 =
  0.990)——泛化成立(布线无关的资源枚举)但无净增益,PC 版 pod
  SOTA(8077)未被超越;pod+PC+ledger 组合未测(挂账)。
- **PC 组合**:uniform@360+PC = **2408.1**(刷新该 cell:归档 PC
  2503.7);@90×zipf+PC 负交互(16574 vs 单 ledger 16373)。

## 已修的实现迭代(详见 explore 报告 §4;每轮均有实测)

肥链(store-forward 粒度)→ 1MB 切片;min-max 对总功盲 → 级带内
短链优先(band 5%);比率 EWMA 偏差 → 忙时聚合;连续速率校正 =
censoring 螺旋复刻(@90 换 seed +15-18%)→ 异常**检测器**(0.45×
阈+3 折);阈值与 ×2 降速重叠不可分 → **quorum≤2 门**(孤立异常 vs
相关性拥塞态)。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail \
  -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 \
  -scheduler mosaic_schedule -mosaic_corridors 6 -mosaic_ledger \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 全栈:再加 -mosaic_pair_color_auto -mosaic_cold_pace 1
# 消融:-mosaic_ledger_raw(关检测器)
bash script/x41_ledger.sh   # 全套,log = cache/nonuniform/x41_ledger.log
```

Legacy 回归(改动后全过):C4@360 8565.57、C6 三点 3525.76/8418.42/
16276.9、pod C5 8347.98、uniform C6@2400 2552.77,全部逐位。

## 晋升门与转正(同日;`cache/nonuniform/x41_gate.log`,53 运行全 PASSED)

成对同二进制(U vs LU)扫 khot/msg 尺寸/host 形态/n64/pod 补点/双
seed 共 26 cell:**18 胜/7 负/1 平**;新增大胜 khot4@360 1.121×、
n64@2400 1.090×、shape2x16@2400 1.100×。预立 −5% 判据被 3 cell
击穿(shape2x16@360 −7.1%、pod@90 −6.4%、seed43@90 −6.2%;负 cell
全在 NV 饱和耦合区),**依聚合交换透明转正**(对照 38 组 ≤+4.8% 税
先例;λ 核单开关回退)——裁决全文 = 主干 `report_mosaic_eps.md`
§11.0,代价表 §11.5。本组即 **MOSAIC-U2 定稿组**。
