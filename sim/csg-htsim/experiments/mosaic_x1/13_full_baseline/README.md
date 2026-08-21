# 13_full_baseline — 非均匀 A2A × 全部 baseline(问题 1)

hot-dst zipf1.0/seed42,flex_rail 32 GPU,nv∈{2400,360,90}。离线 solver 按
铁律在**与执行完全一致的 topology 描述**上重解(TE-CCL topo json 带
nv_gbps 字段;SyCCL config 改 nvlink bw_mbpus;TACCL intra beta 按
2400/nv 重标定 + 32MB padded 重标定),pad-to-max + fill 截断语义
(`tools/fill_schedule_nonuniform.py`)。nv=2400 ≡ dgx_rail,solver 行复用
缓存解。全部 validation PASSED,无一 SKIP。数据 `results.csv`。

## 主表(completion_us;括号 = MOSAIC 相对该方案的加速比)

| nv | bound | ECMP | TE-CCL | TACCL | SyCCL | FAST | **MOSAIC** |
|---|---|---|---|---|---|---|---|
| 2400 | 3327 | 21829(6.1×) | 17450(4.9×) | 17451(4.9×) | 15927(4.5×) | 3541(1.00×) | **3557** |
| 360 | 7556 | 21829(2.5×) | 11762(1.37×) | 15302(1.79×) | 15717(1.83×) | 10126(1.18×) | **8566** |
| 90 | 15105 | 21829(1.33×) | 22640(1.38×) | 31299(1.91×) | 23231(1.42×) | 40479(2.47×) | **16368** |

(MOSAIC 为 C=4 默认;C=6 见 21 组,各点再快 0.5–2%。)

## 结论

1. **MOSAIC 在所有 9 个 (solver, nv) 组合上全胜**,幅度 1.37×–4.9×。
2. **强 scale-up(2400)**:solver 输在 pad-to-max 的计划结构(§1.2 已证:
   热点入口摊不开),4.5–4.9×。
3. **弱 scale-up(360/90)**:solver 的相对位置反而改善(TE-CCL@90 22640
   还比 FAST 40479 好)——**LP 看得见 nv=90 的拓扑**,会少用机内层;
   但 pad-to-max 语义 + 离线时延(见 18 组:分钟级)仍让它输 MOSAIC
   1.37–1.91×。这说明单靠"拓扑感知"或单靠"在线"都不够,需要两者兼备。
4. TACCL@90 的重标定编码(beta×26.7 + 时间单位重标定)可解、未撞哨兵;
   耗时 ~17min(routing+scheduler 各 600s 上限内)。
5. ECMP 在 90 点(21829)好于 FAST(40479)和 TE-CCL(22640):不作为
   反而避免了错误的机内绕行——弱 scale-up 下"绕行要划算"是强约束。

复现:`script/run_dgx_rail.sh`(nv=2400,NONUNIFORM=1)+ 本目录 README 的
flex 变体流程(solver 侧文件在 `cache/nonuniform/{teccl,taccl,syccl}_flex*`,
执行统一 `-topology flex_rail -flex_nv_gbps {nv} -schedule_file <filled>`)。
