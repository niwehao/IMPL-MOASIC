# 38 组:统一方案 MOSAIC-U 全域扫描(35 cell)

**日期**:2026-08-01。**目标**:结束"A 机制在 X 好、B 在 Y 好"的
regime 特调——定稿**唯一统一方案**并在所有历史 SOTA 场景补测。

**MOSAIC-U(EPS)** = `C(wiring 定)+ W(拓扑可调)+
-mosaic_pair_color_auto + -mosaic_cold_pace 1 [+ cube LP 表(离线产物)]`。
PC 的 auto 门本组扩为三重:diag-CoV(34 组)∧ **rail2p wiring 门**(新,
34/35 组结构性负)∧ **lp_spread 门**(新,本组归因 cube mix +3.1% 全来自
PC×jdist 争 lane 时序;`src/main_regular.cpp` 生成期判定)。
R(MOSAIC-R)/HOSTL/E 系机制退役(`Note/fail.md` §1)。
**MOSAIC-O(OCS)** 本就单一方案,无需改动。

## 结果(35 cell;results.csv,原始日志 x38*.log)

- **新 SOTA(10)**:rail 180/420、pod 720、rail2p 2400B4/**360 对称
  (−3.4%)**、cube full300/full50、khot2、**64GPU(−2.3%)**、
  **drift(−1.1%)**。
- **持平/逐位(8)**:rail 360/480/600/720、uniform、persrc、cube
  mix(4003.27 = 已知最优)/250x16、clos zipf/incast。
- **统一代价(10)**:最大 cube 100x16 +4.8%(纯 PACE);rail 2400
  +3.0%(R 退役税,唯一 vs FAST 落后点 0.975×);khot8 +3.3%;
  其余 ≤2.1%。完整清单 `Note/fail.md` §5。

## 回归与方法学

- 代码改动仅 PC auto 门(生成期,两个 else-if);rail nv360 zipf
  7939.33、cube mix 4003.27、uniform 2503.67 与历史逐位一致。
- 64GPU 用 2MB base 强契约 demand(36 组曾把 cache 的 n64 文件重生成为
  weak 契约,本组用临时强契约文件对齐 32 组历史行——引用时注意两种
  契约不可混)。
- 全部 validation PASSED;对比数字逐格给出历史特调配置与差值。

## 复现

```bash
script/x38_unified.sh      # 主扫(rail/pod/rail2p 代表点/cube mix/clos)
script/x38b_unified2.sh    # cube 画像/khot/64GPU/drift
# rail2p 其余 6 cell 与归因 run 的命令见 x38c_rail2p.log 头部与 fail.md
```
