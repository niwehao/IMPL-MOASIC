# 43_ocs_arbiter — MOSAIC-O-A:重配置请求的控制平面仲裁

`htsim_optical -wave -wave_arb`:x40 O-W 的执行器(class-major key 带 +
belt hold + era + 看门狗)重做成显式控制平面请求账本——同边请求合并
(一次 R)、D 请求授予时注入(晚绑定)、charge 前瞻的带序授予、
端口水线准入否决(margin 0.9)、storm 自动 R-quantum。发射侧波/λ/
正典常数逐字不动;默认 `-wave` 与 x40 逐位一致。

- **主网格(n=32,4 demand × 4R)**:vs x40 冻结值 **5 胜 0 负 11
  逐位平**——uniform@10 1.075×、uniform@100 1.054×、zipf@10 1.365×、
  zipf2@10 1.436×(水线否决 x40 自伤的混合准入,贴 LB×1.106)、
  persrc@10 1.290×;forced ≤5(x40 小 R 列 reopens≈3500 的 roll 机器
  被合并吃掉)。
- **漂移 5 点、n64 3 点**:与 x40 逐位相同(era 退役,请求语义等价)。
- **同步**:仲裁器只用 charges(控制器邮箱)+ 交换机本地 served,
  无主机遥测;`-wave_ctl_us` 控制往返税表见 results.csv(R=10 细波
  regime 要求 δ≲2µs,R≥100 容忍 δ≤20µs)。
- 复现:`bash script/optical_arbiter.sh`(日志
  `cache/optical/x43_arbiter.log`,md5 头 + DONE);对手 35/39/40 全
  冻结引用,未重跑。
- 详情与 8 轮失败迭代:`Note/explore/report_ocs_arbiter_x43.md`;
  主干结论:`Note/report_mosaic_ocs.md` Part B。
