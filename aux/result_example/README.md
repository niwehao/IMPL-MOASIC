# result_example —— 各观测面的真实输出样例

每个文件都是**真实运行的原始输出节选**(非手写),用途:①新会话/新人对照
"我看到的输出长这样吗",②debug 时判断观测面本身是否健康(拿到的现场与
样例形状不符 = 先怀疑观测面,再怀疑协议)。样例采集于本机 gpu-02/gpu-01
双机(2026-08-25,统一搬运 API 版本);Raven 样例注明出处 jobid。

| 文件 | 怎么产生(原命令) | 什么情况下出现 | 口径 |
|---|---|---|---|
| `bench_run_line.csv` | `G=4 ./aux/scripts/run2host.sh --msg 1048576 --iters 2 --warmup 1 --sched mosaic --verify 1` | 每次 bench 结束打一行(stdout);`--verify 1` 失败时 med=NaN | `med_us` = rank0 开门时刻中位(warmup 与校验轮不计);makespan 取 `[prank]` 行 per-rank max |
| `phase_lines.txt` | 同上 + `MOSAIC_PHASE=1` | 每个 collective 一行(stderr,WARN 级) | notice=event 线程察觉提交;bcast=demand 行广播完;first_post=首片投递;plan_us=注水 CPU 耗时;evmax=事件循环最慢段(段号:µs)。全部相对 submit 时刻 |
| `xport_profile.txt` | 任意跨机运行 + `MOSAIC_LOG=2`(INFO 级) | CommInit 一次(每 rank);单机形态不出现 | loopback QP 实测,每方向 Gbps,`uni`=单向(上限≈线速)/`bi`=双向(暴露排水劣化);→ 后为选中组合。**共享 HCA 的多 rank 并发探测会均分带宽**(本机 4 rank ⇒ 每 rank ≈ 1/4 线速),相对比较仍有效 |
| `trace_slice.csv` | 同 bench + `MOSAIC_TRACE=1`;后处理 `python3 aux/tools/trace2csv.py /tmp/mosaic_rank0.trace` | trace ring(mmap,进程崩溃也留最近窗口)恒在写,默认关 | 32B 定长事件;SLICE_POST/RECV/DELIV = 单片生命周期;跨节点比对时间轴须先对 NTP(CLOCK_MONOTONIC 跨机有 ~11.6ms 偏移,log_raven.md) |
| `moe_loss_gate.txt` | `MT_BACKEND={nccl,mosaic} ITERS=60 bash mosaic-rt/nccl/run2.sh moe-rt/train_moe_mcore.py`,两腿 `iter N loss` 整列 md5 | 公平性门(bit-exact 铁律):md5 逐位一致才算过 | 只认 `[r0]` 行整列(局部 loss 每 rank 不同,裸 grep 首行必出假分歧,log_raven.md 29345490) |
| `dump_state.txt` | 无进展 2s(watchdog)或 SIGTERM 自动落 `/tmp/mosaic_dump_rank<r>.txt` | **只在卡死/被杀时出现**;正常 PASS 的 run 里是空文件 | 样例出自 Raven 29353960-162453(旧字段版:`q ... tx= rx=`;现行版本为 `q nv= nic= kq=`,kdesc 队列取代了 CE 在途队列)。`recv x/y` 两数不等 = 谁没收齐;`src N recv/want` 逐源对账 |
| `raven_verdict.txt` | Raven sbatch 模板尾部自动打(出自 29455196 MoE 进表作业) | 每个 Raven 作业必有;`rwatch.sh` 只看这行判 PASS/FAIL | 形态核验行(4×4 同档)在前;`VERDICT: PASS|FAIL` 一行结论,不要人肉翻日志 |

维护约定:观测格式变更时同步更新对应样例(用真实输出覆盖,不手改)。
