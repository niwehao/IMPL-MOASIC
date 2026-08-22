# 模拟器使用手册(怎么把这套东西跑起来)

这份文档回答一个问题:**拿到这个仓库,怎么编译、怎么跑一个实验、
结果去哪看**。设计与结论看 `Note/report_mosaic_eps.md`(主干);
更细的操作坑清单看同目录 `handoff.md`。

## 0. 这套仿真是什么

一个包级仿真器,在 `csg-htsim/` 下:

| | 说明 |
|---|---|
| 可执行 | `sim/tests/htsim_smart_regular` |
| 粒度 | 包级(1KB 包,共享队列排队) |
| 比什么 | MOSAIC vs FAST/ECMP/三家离线求解器 |
| 一次运行 | 一个 all-to-all collective 从发起到收完 |
| 输出 | 一行 `validation PASSED ... completion_us X`(X 越小越好) |

**铁规矩**:只有 `PASSED` 的数字才算数(表示每个字节都送达且对账
一致);`FAILED` 只能用来诊断。

## 1. 编译

```bash
cd csg-htsim
make -C sim/tests htsim_smart_regular
```
全部 header-only,改任何 `.h` 都要重编;没有 error 字样即成功。

## 2. 跑一个实验

```bash
D=cache/nonuniform/demand_n32_zipf1.0_seed42.txt   # 流量矩阵文件
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail \
  -flex_nv_gbps 360 -packet_size 1024 -chunk_size 33554432 \
  -scheduler mosaic_schedule -mosaic_corridors 6 \
  -mosaic_ledger -mosaic_pair_color_auto -mosaic_cold_pace 1 \
  -workload A2A -all_to_all -store_forward -quiet -demand_file $D
```

逐个旋钮说人话:
- `-flex_nv_gbps`:每块 GPU 的机内高速通道带宽(实验主变量,越小
  代表机内越弱;180=网卡同速,90=只有网卡一半);
- `-chunk_size`:非均匀流量时抄 demand 文件头部的
  `padded_pow2_bytes`;均匀流量时删掉 `-demand_file`、改用
  `-message_size 2097152 -chunk_size 2097152`(每对 2MB 的契约);
- 第三行三个 flag 就是**现行 SOTA 配置(MOSAIC-U2)**:账本核 +
  配对着色 + 冷源限速,照抄即可;
- 换对手:`-scheduler fast_schedule -fast_alpha_beta_guard`(FAST,
  guard 必带)或 `-scheduler static_ecmp`(基准);
- 想知道理论最快:`python3 tools/x1_bounds.py --demand $D --nv-gbps
  360`,输出 `bound_us`,你的 completion 除以它就是"距界"。

## 3. 对比新方案 = 基准闸一键(不要手工逐个跑)

```bash
NAME=myscheme SCHED=mosaic_schedule FLAGS="-mosaic_ledger -my_flag" \
  bash script/bench_scheme.sh
```
它会把标准 67 个测点(主线 8 个带宽点/换种子/换偏度/均匀/多热点/
消息尺寸/机器形态/64 卡/pod 布线/网卡降速/交换机瘦身/过期先验)
全部跑一遍,然后自动和**缓存好的基线**(理论界/ECMP/FAST/旧
MOSAIC/现 SOTA)拼成对比表:

- 每 cell 的倍数表打在屏幕上,CSV 落在
  `experiments/mosaic_x1/bench_runs/<NAME>.csv`;
- **基线永远不重跑**(缓存在 `experiments/mosaic_x1/
  bench_baselines.csv`);你的新方案要成为以后的对照,跑完执行
  `python3 tools/bench_compare.py <NAME> --promote`;
- 想宣称新 SOTA:每 cell 不差于现 SOTA 5% 以上且零 FAILED(判据与
  例外流程见 CLAUDE.md"晋升门")。

## 4. 生成流量矩阵

```bash
python3 tools/gen_nonuniform_a2a.py --nodes 32 --message-bytes 2097152 \
  --alpha 1.0 --seed 42 --output cache/nonuniform/demand_n32_zipf1.0_seed42.txt
# --mode hotdst(默认,热门目的地)| persrc(每源独立)| khot --khot K
```
每个源的总量恒等于均匀契约(公平可比);常用矩阵都已在
`cache/nonuniform/` 里,先找再生成。

## 5. 结果与文档去哪看

- 每组实验:`csg-htsim/experiments/mosaic_x1/<编号_名字>/`
  (results.csv + plot.png + README,总索引在上层 README);
- 原始批日志:`csg-htsim/cache/nonuniform/`;
- 结论与设计:主干报告 `Note/report_mosaic_eps.md`;探索过程/失败
  记录:`Note/explore/`。

## 6. 最容易踩的五个坑

1. 忘了 `PASSED` 才算数——FAILED 常见原因是 buffer 溢出丢包
   (默认 1GB 无重传),大消息/大规模先想 backlog,必要时
   `-flex_fabric_buffer_mb` 加大并把溢出本身当发现记下来;
2. 非均匀实验 `-chunk_size` 没抄 demand 文件头 → 对不上历史数字;
3. 跑离线求解器(TE-CCL/TACCL/SyCCL)不看缓存直接重解——`.sched`
   在 cache 里就直接用,`FORCE_RESOLVE=1` 才强制重解(TACCL 还必须
   cd 到它的 cache 目录再跑,见 handoff);
4. 比 FAST 忘带 `-fast_alpha_beta_guard`(不公平,结论无效);
5. 长批次开了双层后台(脚本里再 `&`)——父 shell 退出会把任务
   杀掉;一层后台 + 日志末尾 DONE 标记即可。
