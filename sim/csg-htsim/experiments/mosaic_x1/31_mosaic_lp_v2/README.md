# 31_mosaic_lp_v2 — FAST 差距归因 + LP-spread 统一(jdist/hot-column)

2026-07-27。两问:①同构点对点上 MOSAIC 为何不收敛到 FAST,能否**不检测
同构、不感知劣势**地自动收敛;②LP 能否覆盖 NVSwitch(flex_rail),
用 MOSAIC-LP 统一全部 wiring。完整叙事:`Note/explore/2026-0727-x31-mosaic-lp-v2.md`。

## 差距归因(diag 段,x31_diag.log)

- **主因 = spread lane 落点集中**:同构下"按速度排序"退化为按 id 排序,
  所有 pair 的 lane-0 砸同一 NIC。证据:full300 仅 lane_rot 就
  4754→3630(FAST 3660);uniform demand 的 full300 同样劣(3009 vs
  FAST 2344),不是 skew 特有。反馈无罪(static 更差 6535)。
- 残余 ~5-8%(250x16、uniform-demand 同构)是 M1(逐对置换协调),
  spread 层解不了。

## 机制演化(jdist 三步,都 workload-free)

| 版本 | full300 | 400x16 | 教训 |
|---|---|---|---|
| 顶点解(每 pair 单轨,全局 Latin 均衡) | 4906 ✗ | 5186 ✗ | 均匀先验的单轨指派在 skew 下成 per-pair 热点 |
| +多样性下限 div_k=4(单轨 ≤1/4) | 3816 | 4200 | 构造性 skew 鲁棒 |
| div_k=6(全轨) | **3554** | **3531** | 同构点反超 FAST(3660/3809) |
| hot-column LP(逐 rd 全轨涌入先验,NIC+机内链路同计价,ε·距离保局部性) | **3520** | 3587 | 异构容量感知,mix/seed/zipf 大赢 |

## 主表(全部 PASSED;stack = C6 + W12/16 + lp_intra(+local) + jh-spread)

| 配置 | bound | FAST | 30 组 SOTA | **31 组 stack** | vs FAST |
|---|---|---|---|---|---|
| mix 主线 | 3327 | 5402 | 4624 | **4134.6**(1.243×) | **1.31×** |
| 400x16 | 3327 | 3809 | 4308 | **3531**(jd_k6,1.061×) | **1.08×** |
| full300 | 3327 | 3660 | 4754* | **3520**(1.058×) | **1.04×** |
| full50 | 7695 | 10710 | 10328* | **8932**(1.161×) | **1.20×** |
| 250x16 | 3456 | **4399** | 4766 | 4542(1.314×) | 0.97×(FAST 仍胜 3.3%) |
| 100x16 | 7031 | 10920 | 8094 | **8045**(1.144×) | 1.36× |
| 50x16 | 10732 | 21839 | 12003 | 12066(≈) | 1.81× |
| rot4/8/12 | 3776/4161/3327 | 5800/5620/5509 | 4994/5215/4293 | 5161/5534/5273 | 1.12/1.02/1.04× |
| seed43 | 4048 | – | 4818 | **4354**(1.076×) | – |
| zipf1.5 | 5269 | – | 7326 | **6324**(1.200×) | – |
| uniform demand | 2237 | 2456-2344 | 2591 | 2591(skew 门关,逐位=base) | 0.95×(FAST 胜,M1) |
| rail nv90/360/2400(jd 星图) | 15104/7552/3327 | – | base 16277/8418/3526 | 16246/8373/3511(−0.2~0.5%) | – |

\* 29 组 base(当时无该点专用优化)。

- **统一配置在除 250x16/uniform-homo 外全部测点 ≥ FAST**,且无同构
  检测——同构行为是 LP 在均匀容量下的自然解,不是分支。
- **诚实记录**:rot4/8/12 上统一 stack 不如 30 组各自专配(rot12 5273
  vs 4293;jh 的"全轨涌入"先验在 hot 落极弱 B_in 时仍偏均匀);
  rail 上 jdist 与 R 机制组合为负(8258 vs R 7970),rail 推荐仍是
  MOSAIC-R,LP-spread 对 rail base 中性偏正(−0.2~0.5%)。
- uniform demand 的 −4% 局部性代价由 **skew 门**(与 E3/F3b 同源,
  demand 已知非拓扑检测)消除:门关逐位=base。
- 回归:rail C4@360=8565.57、cube W16 base=4824.19 逐位不变。

## 复现

```bash
P=/home/weihaoni/miniconda3/envs/myenv/bin/python3
$P tools/mosaic_lp_intra.py --mode jdist_hot --speeds 400x6,200x6,100x4 \
  --output cache/mosaic_lp/jh_mix_rot0.txt     # ~26ms
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_cube \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -mosaic_waves 12 \
  -mosaic_lp_intra cache/mosaic_lp/cube_mix_rot0.txt -mosaic_lp_local \
  -mosaic_lp_spread cache/mosaic_lp/jh_mix_rot0.txt \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 同构消融表:--mode jdist --div-k {1..6};rail 星图:--mode jdist_hot --nv-gbps NV
# 批日志:cache/nonuniform/x31_{diag,lpv2}.log、x31b.log
```
