# 30_mosaic_lp — MOSAIC-LP:固定拓扑=LP,不固定 workload=动态调整

2026-07-27。承接 29 组(flex_cube,SOTA=C6+W16/R)与用户思路:**拓扑
到手时(workload 未知)用 LP 离线解子调度,运行期用 MOSAIC 的 λ 反馈
调子调度比例**;LP 下放到机内(逐 host 小 LP),跨级只做比例调整,
不维护全局大 LP。完整叙事:`Note/explore/report_mosaic_lp.md`。

## 机制

- `tools/mosaic_lp_intra.py`:对每个本地 dst d 解一个机内最小拥塞
  多商品流 LP(每个顶点 j→d 单位商品同时活跃 = workload-free 均匀
  先验;TE-CCL 式 mcf 内核,时间维坍缩为速率域),路径分解(≤3 链)。
  **8 GPU × 8 个 LP 共 32–48ms/拓扑**;同构 host 共享一张表。
- htsim `-mosaic_lp_intra <表> [-mosaic_lp_local]`:RX/TX 的机内段
  (及 lp_local 时同机 chain)按 LP 路径比例拆多条 chain;λ 仍按
  corridor 调 → **LP 管空间(机内字节走哪),运行期管时间(每波
  每走廊多少)**。

## 主表(全部 PASSED;Δ 相对 29 组每档 SOTA)

| 配置 | SOTA(29 组) | +LP(_local) | Δ | vs bound |
|---|---|---|---|---|
| mix rot0(W12+lploc 最优) | W16 4824 | **4624** | **−4.1%** | 1.390× |
| mix rot4 | W16 5232 | 4994 | −4.6% | 1.322× |
| mix rot8 | W16 5511 | 5215(lp,不带 local) | −5.4% | 1.253× |
| mix rot12 | W16 4717 | **4293** | **−9.0%** | 1.290× |
| 100x16 | W16 8891 / R 8306 | **8094**(W16+lploc) | −8.97%/−2.6% | 1.151× |
| 50x16 | R 12809 | **12003**(R+lploc) | −6.3% | **1.118×** |
| 250x16 | W6 4917 | 4766 | −3.1% | 1.379× |
| 400x16 | W12 4308 | 4328 | +0.5%(中性) | 1.301× |
| full300(退化) | 4753.77 | 4753.77 | **逐位相同**(LP 全直连=无操作,sanity) | 1.429× |
| uniform demand | 2591 | 2526 | −2.5% | 1.129× |
| seed43 | W16 5224 | 4818 | −7.8% | 1.190× |
| zipf1.5 | W16 8219 | 7326 | −10.9% | 1.390× |

- **无一场景劣化超过 +0.5%**;异构/稀缺档一致 −3~11%。
- W 交互(mainline lploc):W6 4895 / W12 **4624** / W16 4679 / W24 4652
  ——LP 多路径吸收了一部分细波的作用,最优 W 从 16 回落到 12。
- 400x16(同构快档)LP 无路可分(直连或等价对称),中性;该档
  FAST(3809)仍领先 —— M1 配对协调缺失,不是 LP 能解的。
- flex_rail 8 nv 点 C6 回归逐位一致(emit_chain 重构后复验,
  x29_reg.log)。

## 复现

```bash
cd csg-htsim && make -C sim/tests htsim_smart_regular
P=/home/weihaoni/miniconda3/envs/myenv/bin/python3
$P tools/mosaic_lp_intra.py --speeds 400x6,200x6,100x4 --output cache/mosaic_lp/cube_mix_rot0.txt
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_cube \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -mosaic_waves 12 \
  -mosaic_lp_intra cache/mosaic_lp/cube_mix_rot0.txt -mosaic_lp_local \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
# 批日志:cache/nonuniform/x30_lp.log(DONE 结尾);LP 表:cache/mosaic_lp/
```
