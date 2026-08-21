# MOSAIC × FAST 实验总集(flex_rail / X1,2026-07-10,v5 扩充 2026-07-11)

配套报告:`Note/report_mosaic_eps.md`。每个子目录一个实验:`results.csv`(数据表)
+ `plot.png`(图)+ `README.md`(设置与一句话结论)。复现:01-12/16-21 均为
`sim/tests/htsim_smart_regular -topology flex_rail ...`(参数见各 README 与报告 §9),
demand 矩阵在 `cache/nonuniform/`;13/15/18 含离线 solver 行(缓存/重解流程见各
README,TACCL 按约定 cd 到 cache 目录求解)。

| 目录 | 实验 | 一句话结论 |
|---|---|---|
| 01_crossover | 主交叉线(8 个 NV:NIC 点 × 方案 × 下界) | 交叉点 ≈3.5:1;弱侧 MOSAIC 1.05–2.47×,强侧 ±1% |
| 02_ablation | 先验×反馈×走廊混合消融 | 反馈强侧+3~14%、坏先验保险 2.5×;单摊开走廊崩 2.6× |
| 03_fairness_fast | FAST 公平性(guard/轮询/scaling) | 弱侧塌陷映射无关,scaling 无救 |
| 04_robustness | seed×α×workload 家族 | 弱侧 2.1–2.65× 全稳;强侧 ±10% 矩阵依赖 |
| 05_drift | 陈旧先验(persrc) | adapt 把 stale 偏差从 ±8% 压到 ≤3% |
| 06_uniform | 均匀 A2A sanity | nv=90 连均匀都赢 1.17×;2400 慢 9%(FAST 主场) |
| 07_scale_n64 | 64 GPU 规模 | 1GB 缓冲下 ECMP/FAST(nv90) 溢出 FAILED、MOSAIC 波次=隐式流控全 PASSED;4GB 缓冲公平版另列 |
| 08_param_sensitivity | W/γ/threshold 敏感性 | 全参数带宽 ~10%,默认无需逐点精调 |
| 09_multihot | K∈{2,4,8} 等热多热点 | 全部 MOSAIC 赢(360: 1.18–1.27×;90: 2.1–2.5×) |
| 10_msg_size | 集合尺寸 0.25×/4× | 见 CSV/图 |
| 11_taper | spine taper {90,180,360} | 见 CSV/图 |
| 12_lambda_trace | 热 GPU λ 收敛轨迹 | 2400 收敛到 RX 重、90 到 DIRECT/TX 重(水填充两端) |
| 13_full_baseline | 非均匀 × 全 baseline(含 TE-CCL/TACCL/SyCCL 按各 nv 重解) | 拓扑感知 LP 也躲不开 pad-to-max:solver 全点输 MOSAIC 1.4–4.9× |
| 14_uniform_congestion | 均匀 A2A × taper/ECN | 无拥塞逼平 FAST(−2~−10%);taper 只罚 ECMP;skew 才是拥塞战场 |
| 15_skew_sweep | zipf α∈{0.5..2} × 全方案 | α=2 时 2400 强侧也反超 FAST 1.18×;α=2@90 距界 +0.4% |
| 16_corridor_count | corridor 数消融 C∈{1..6} | C=1≡ECMP;结构完备性>数量;C=6 全点微优于 C=4 |
| 17_threshold | 波阈值 {0.3..0.95} | 全带宽 ≤6%,0.75 非精调;0.95 弱支配 |
| 18_synthesis_speed | 生成速度 vs 规模(32→256 GPU) | MOSAIC µs 级 & 平缓;FAST ms 级超线性;solver 分钟级起步 |
| 19_degraded_nic | 未通告 NIC 降速(热/冷 × 2/4×) | 热点降速让 2400 平手变 MOSAIC 赢 1.44×;执行期不确定性改变 crossover |
| 20_host_shape | 8h×4g / 2h×16g 形态 | crossover 形状两个极端形态都成立;2h16g@2400 赢 1.09× |
| 21_design_iter | 设计迭代:C=6 (+thr) 联合验证 | **C=6 全 8 nv 点优于 C=4 且 vs FAST 全点 ≥1.00×(至 2.49×)** |
| 22_x2_pod | X2:pod 布线+4 spine,corridor 族再生成 | 全 9 点赢 FAST 1.19–2.59×,多点贴割界;prior 公式零改动 |
| 23_x3_rail2p | X3:双平面 rail + 平面不对称 | 360/90 全档赢;发现奇偶盲区(同轨中继恒钉 A 面),2400+B/4 缺 B 面中继类(如实) |
| 24_x4_threepipe | X4:三管 corridor 族 + K-管注水 prior | 修复奇偶盲区(0.73×→1.02×);known prior 弱 NV 处值 22% |
| 25_x2full_spine | X2-full:显式/隐式 spine 钉法 | 规则 G%S==0 隐式否则显式;FAST 偶数 S 只用一半 spine |
| 26_donation | D2 捐赠/skew 优先序执行期探索(E1/E2/E3/E5) | 尾段已贴界,损失在争用期;skew 优先序+事件锚定每 nv 快 3–5%,gap 最低压到 +1.9% |
| 27_phase_plan | E4 流体相位计划 | 份额模型被否(max-min 子流计数偏差);序预测 Spearman 0.80、last-2 全中 → 计划的正确载体是优先序 |
| 28_wave_ramp | 动态 corridor ramp + 固定 wave size(F1/F2/F3) | cold-restrict ramp(双门)@180 gap +2.0%、MOSAIC-R 全点支配 MOSAIC-P;ramp-hot 与固定 size 证伪(后者仅 360 小赢) |
| 29_flex_cube | 机内 cube-mesh + NVLink 三档异构(M2 边界实测,新拓扑 flex_cube) | MOSAIC WIRING_CUBE 可跑:主线 W16 4824(FAST 5402);NV 稀缺/异构侧 MOSAIC 胜、同构富裕侧 FAST 胜;solver 全 >2.4× 慢;SyCCL 不可表达 |
| 30_mosaic_lp | MOSAIC-LP:机内 35ms workload-free LP + 运行期 λ 调比例 | 异构/稀缺档再 −3~11%、无一劣化;主线新最好 4624(vs FAST 1.17×);50x16 达 1.118× bound;退化档逐位无操作 |
| 31_mosaic_lp_v2 | FAST 差距归因 + hot-column jdist LP 统一摊布(零同构检测) | 差距主因=lane 落点集中;统一 stack 除 250x16/uniform-homo(M1)外全点 ≥FAST;主线 4134.6(1.243×);同构点反超 FAST;rail 星图 LP 中性偏正 |
| 32_paradox_pacing | 带宽/降速双悖论归因 + F6 冷源限速 + F7 per-host λ + 挂账结案 | 23 组重跑(九点全 ≥FAST,0.73× 作废);taper 悖论=冷流量挤占关键路径(solo 单调);F6 三门限速 rail 7/8 点刷新 SOTA;单中继降速 +32% 由 F7 修到 +0.6%;explicit-spine 可达性门 |
| 33_fast_sync | FAST 运行时 stage 同步(`-fast_sync` none/credit/barrier)+ largest-first stage 序修正 | 忠实 credit 语义 zipf 仅 +2.3~3.6%(2400 例外 +11.5%)、barrier +12~14%、uniform ≤4.3% → 既有 async 引用方向保守;desc 序使同步模式 −2.6~−8.2%,async@2400 3541→3419(SOTA 2400 行 vs FAST 收窄至 1.004×) |

画图:`/home/weihaoni/miniconda3/envs/myenv/bin/python3 plot_all.py`
| 34_pair_color | O1 波内配对着色 `-mosaic_pair_color_auto`(Cayley 对角 BvN 化)+ O3 λ 守卫 | M1 翻盘:uniform@360 1.047× 反超 FAST;480/600/720/90 四点刷 SOTA;PC×R 负交互(E3 归因);O3 判负(逐位=base) |
| 35_optical_reconfig | 光电路交换基座 + RoN/Harvest 复现 + MOSAIC-O 三方对比(k=1/k=2、漂移) | MOSAIC-O 20/20 cell ≥ min(RoN 31-d oracle, Harvest θ-opt);k2 定量复现"度数↑重配价值衰减";运行中漂移合并 1.46-1.53× |
| 36_scale | 超大规模 weak scaling 256-1024(EPS+OCS)+ 仿真器桶索引优化(逐位回归) | MOSAIC+PC 距界 1.024×/1.065×;FAST/ECMP 默认 buffer 必炸;FAST per-A2A 构造 111ms→2.70s vs MOSAIC 20ms(85-343×);OCS R=1ms M-O 胜 1.42-1.46× |
| 37_switch_centric | 纯 clos incast/oversub/collision + OCS m-sink incast(跨侧同线速) | incast completion 全员贴割界,差异在 buffer 轴:MOSAIC min-buffer=C/W(wave 定律三连验证)vs blast 0.9C;OCS 静态端点=(m+1)/2×割界(四点全中);M-O 逐点 ≤ min(端点, oracle) |
| 38_unified | 统一方案 MOSAIC-U 全域扫描(35 cell)+ PC 三重自动门 | 10 新 SOTA(rail2p 360sym −3.4%、64GPU −2.3%、drift −1.1% 等)/ 8 持平 / 10 代价(最大 +4.8%);R/HOSTL 退役;PC×lp_spread 冲突发现并门控 |
| 39_partial_reconf | OCS 端口级部分重配(`-async`)+ MOSAIC-O-PR(选择器 + `-pr_online` 在线 duty 反馈)vs 35 组 barrier 归档 | 窗口=中大 R×skew:n32 1.16-1.30×、n64 三 cell 1.20-2.11×;漂移局部吸收 1.14-1.29×(晚期 crossover 门);归因=去 barrier 使能的结构重选(双向环 d=2),持久 skeleton/漏斗证负;v3 5.7-7ms 证伪(Cayley 界 7363) |
| 40_wave_ocs | **MOSAIC-O-W**(`htsim_optical -wave`):EPS wave/λ 方案照搬到 OCS,wave 波次触发端口级重配;零规划,对手=35/39 冻结归档 | x39 窗口在线复刻:zipf@1000 1.29×/持平 PR、persrc@1000 1.33×/**反超 PR 1.10×**;B-gate 大 R 自动单环(unif@10000 1.006×);漂移 4 注入点 13.0-15.0ms、**pr_online 晚期 crossover 治愈(1.36×)**;W 税(小 R 0.43-0.65×,W1 回 RoN 形态)与 shift 缺失(中 R)如实;n64 zipf@1000 1.92× |
| 41_ledger | **MOSAIC-L**(`-mosaic_ledger`):EPS 立足点审计裁决——λ 学习机器(λ/censoring/trust/anti-windup/floor)→ 全局资源账本注水 + NIC 域 quorum 异常检测器;对手=21/38/19/14/22 冻结归档 | 核对核 7/8 胜、栈对栈(LU=+PC+PACE)7/8 ≥ 38 组 SOTA、**2400 vs FAST 翻正 1.002×**、中段距界 480 → +1.2%;**降速 8/8 胜(×4 2.32×/2.58×,距降速界 +4.3%)**;uniform+taper 全胜、uniform@360/90 无 PC 反超 FAST;漂移逐位免疫;**@90×hot-dst −2~−6% 唯一残余**;6 轮迭代(肥链/级带/聚合窗/检测器/quorum/NIC 域) |
| 43_ocs_arbiter | **MOSAIC-O-A**(`htsim_optical -wave -wave_arb`):OCS 重配请求的控制平面仲裁——同边请求合并(一次 R)、D 晚绑定、charge 前瞻带序授予、端口水线准入(margin 0.9)+ storm 自动 R-quantum;对手=35/39/40 冻结归档 | 主网格 **5 胜 0 负 11 逐位平**:uniform@10 1.08×、uniform@100 1.05×、zipf@10 **1.37×**、zipf2@10 **1.44×(否决 x40 自伤混合,贴 LB×1.106)**、persrc@10 1.29×;漂移 5 点 + n64 3 点逐位;era/key 带/绝对 hold 退役;δ 敏感度:R=10 需 δ≲2µs、R≥100 容忍 20µs;8 轮迭代 |
| (bench_runs/u2bl) | **MOSAIC-U2b**(x44,`-mosaic_lg_bcast/-blag_us/-stagger_us/-spineagg`):账本分布式协议重构——重放被 fork 探针证伪(11-39%/波),改双通道广播 + 决策错峰;spine 细分账声明 fabric 前提,ECMP 兜底档价目实测 | 67 cell:vs U2r(重放参考)聚合 −0.13%、4 击穿透明列;**vs λ 39 胜/18 负 +5.0%**;m512k@90 0.80→1.104 修复;通道 B 0.3/1.0/2.6KB(n32/64/256);已 promote 为 `sota_u2`,旧行降 `u2r_replay_ref` |
| 45_spectrum | **MOSAIC-O-S**(`htsim_optical -spectrum`,x45):伙伴数谱系元调度器——四族结构表(M 匹配剥离/S 收缩与 rot/R2 双向环/R1 静态 θ 环)+ 记账水线剪枝 + ≤5 次亚秒干跑(波机 O-A 收编为候选)+ 1% margin 门 + 表侧漂移差分合并;对手=35/39/40/43 冻结归档 | **非均匀 12 判据格全 ≥1.0× vs barrier best**(小 R 全修复:persrc@10 **1.097× 超 PR**、zipf@10 1.003×、zipf2@10 1.023×;@10ms 四格逐位);漂移 4 点=x40 归档(1.11-1.28×);n64 zipf@1000 **2.156×/超 PR 2.0%**;墙钟 n32 ≤0.7s、n64 ≤8.5s(vs A 97.6s);4 轮失败迭代(纯账证伪/raise 死锁/幽灵会话/尾字节环) |
