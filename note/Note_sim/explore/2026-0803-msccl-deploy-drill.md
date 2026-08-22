# 部署日记:把 MOSAIC-EPS 用 MSCCL 搬上真机(虚构演练,第一人称,可照做版)

> **性质声明**:这是一份**假想部署演练日记**(纸上推演,未真实执行)。
> 硬件、报错、数字均为编造,但刻意对齐两样东西:①本仓库 MOSAIC-EPS
> 的方案语义(corridor / waterfilling prior / wave / PC(pair-color)/
> PACE,见 `../report_mosaic_eps.md`);②真实 MSCCL 生态的实现方式
> (microsoft/msccl = NCCL fork 运行时、msccl-tools = MSCCLang DSL→XML、
> msccl-tests-nccl;XML 的 gpu/tb/step/chan 语义、per-connection FIFO、
> 协议约束等)。每天记录**改了哪些文件/函数、为什么改**,目标是照着
> 日记能独立复现。文末附"面试官三轮拷问"自审与复现清单。

**环境设定**:4 节点 × 8× A100-SXM4-80GB(节点内 NVSwitch,NV12),
每 GPU 一块 ConnectX-6 HDR 200Gb/s(8 rail/节点),两层 leaf-spine,
纯 IB。软件:CUDA 12.2(后退回 11.8 容器)、NCCL 2.18.3(基线)、
msccl fork(基于 NCCL 2.12.12)、msccl-tools master、Python 3.10、
nsight-systems、dcgmi、OpenMPI 4.1。
**目标**:32-GPU 非均匀 All-to-All(MoE dispatch/combine),契约对齐
仿真:per-src 总量 62MB(31×2MB),zipf1.0/seed42 热列;基线 = ①NCCL
alltoall(均匀)②grouped ncclSend/Recv 模拟 alltoallv ③pad-to-max。

**我的工程目录**(第 1 天建好,后面所有改动都落在这里):

```
mosaic-msccl/
├── gen/                    # planner:demand → MSCCLang → XML
│   ├── waterfill.py        # 注水先验(抄 scheduler/mosaic_schedule.h 语义)
│   ├── corridors.py        # DIRECT/TX/RX/SPREAD3 劈分 + rail 落点
│   ├── waves.py            # wave 切分 + scratch 槽位分配
│   ├── epoch_order.py      # PC 着色 + 连接内全局键序(核心不变量)
│   ├── emit_mscclang.py    # 生成 msccl-tools DSL 程序并调用 XML()/Check()
│   └── check_dag.py        # 自写静态验环器(networkx)
├── bench/
│   ├── bench_a2av.cu       # grouped send/recv 基线 + msccl 路径 + 校验
│   ├── patterns.h          # buffer 填充/校验模式
│   └── run_grid.sh         # 扫参脚本(NCCL_* 环境网格)
├── patches/                # 对 msccl fork 的小补丁(git format-patch)
├── profiles/               # 12 格 XML 库 + 元数据 json
├── torch_ext/              # torch 侧胶水(repack kernel + ncclAllToAll 绑定)
└── scripts/                # nsys/counters/长跑脚本
```

---

## 第 1 周:环境、基线、msccl 编译

### 8月3日(周一)
拿到配额。硬件核对三件套(以后每次怀疑硬件先跑这三个):
```bash
nvidia-smi topo -m                  # 期望 8 卡全 NV12
ibstat | grep -E "State|Rate"      # 期望 8 口 Active / 200
ibdev2netdev                        # mlx5_0..7 与网口映射,记下来
```
基线锚点(nccl-tests):
```bash
mpirun -np 32 -hostfile hosts --map-by ppr:8:node \
  -x NCCL_DEBUG=WARN ./build/alltoall_perf -b 62M -e 62M -g 1 -n 50 -w 20
```
均匀 alltoall **3.42ms**(algbw 18.1GB/s)。手算下界:跨机 46.5MB ÷
25GB/s ≈ **1.86ms**,NCCL 距界 1.84×。
**改动**:把仓库 `tools/x1_bounds.py` 抄成 `gen/bounds.py`,输入改为
(demand, nic_GBps=25, nvlink_GBps=300, ranks_per_node=8),输出
egress/ingress/per-rail 三个割;后面所有表格的 LB 列都出自它。

### 8月4日
**改动**:写 `bench/bench_a2av.cu`(约 400 行),这是整个项目的测量
底座,结构:
1. `fill_pattern()`:每个 sendbuf 元素填 `(rank<<24)|(peer<<16)|idx`
   (`patterns.h`),收端 `verify()` 逐字对——**先有校验再谈性能**;
2. alltoallv 基线路径:
   ```c
   ncclGroupStart();
   for (int p = 0; p < 32; p++) {
     if (send_bytes[p]) ncclSend(sptr[p], send_bytes[p], ncclInt8, p, comm, s);
     if (recv_bytes[p]) ncclRecv(rptr[p], recv_bytes[p], ncclInt8, p, comm, s);
   }
   ncclGroupEnd();
   ```
3. 计时:warmup 20 + 50 次,`cudaEventRecord` 夹整个 group,汇报
   mean/p95(`MPI_Allreduce` 取全 rank max 再统计);
4. demand 从文件读(直接复用仓库 `cache/nonuniform/demand_n32_*.txt`
   的格式,`demand s d bytes` 三列——仿真和真机用同一份 demand)。
结果:grouped send/recv(zipf)**28.6ms**(LB=热列 508MB÷25=20.3ms,
1.41×);pad-to-max alltoall **41.6ms**(每对 pad 到 16MB)。

### 8月5日
编 msccl。CUDA 12.2 下直接炸(ptxas `__assertfail`、gcc-12 模板报错)。
**决定**:不给老代码打现代化补丁,起 `nvcr.io/nvidia/pytorch:22.12`
(CUDA 11.8 + gcc-11)容器:
```bash
git clone https://github.com/microsoft/msccl && cd msccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# 产物 build/lib/libnccl.so —— 就是"带 MSCCL 解释器的 NCCL"
git clone https://github.com/Azure/msccl-tools && pip install -e msccl-tools
git clone https://github.com/Azure/msccl-tests-nccl  # 带 alltoall 测试
```
**注意**:bench 里链接哪个 libnccl 决定走哪个栈——`LD_LIBRARY_PATH`
指到 msccl 的 build/lib 即可,API 与 NCCL 完全兼容(msccl 是 fork)。

### 8月6日
跑通官方示例,搞清"算法怎么被选中":
```bash
python msccl-tools/examples/mscclang/alltoall_allpairs.py 32 1 > allpairs.xml
mpirun ... -x LD_LIBRARY_PATH=$MSCCL/build/lib \
  -x MSCCL_XML_FILES=allpairs.xml -x NCCL_ALGO=MSCCL,RING,TREE \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT ./bench_a2av --uniform
```
必须看到这行才算加载:
`NCCL INFO MSCCL: Parsed algorithm allpairs ... nchannels 1 nchunksperloop 32`。
**第一坑**:加载了但没走 MSCCL(耗时=NCCL)。原因:msccl 在 enqueue
时要求 `count % nchunksperloop == 0` 才会命中自定义算法,我的
62MB/32rank 除不尽。**改动**:bench 里把每对消息 pad 到 2MiB 整——
granularity 契约在真机的第一次具象化。再跑 allpairs:4.96ms(1 chan,
慢是正常的,通路验证完成)。

### 8月7日
读运行时源码日(不改代码,做笔记)。重点文件:
- `src/graph/mscclxml.cc`:XML 解析,各字段的合法域;
- `src/collectives/device/msccl_kernel.cu`(解释器):tb 循环取 step,
  `s/r/rcs/rrc/cpy/nop` 的执行,`depid/deps/hasdep` 如何实现跨 tb 等待
  (volatile flag 数组);
- `src/enqueue.cc`:算法选择逻辑(collective 类型 + 整除性)。
**贴墙硬约束清单**(后面设计全部围着它转):
1. 一个 `<tb>` 只绑一个 (send peer, recv peer, chan) 三元组;
2. 同一 (peer,chan) 的连接是 **FIFO**:两端 step 序必须逐元素匹配;
3. `nchannels ≤ 32`;steps/tb ≤ 256(我们版本);`s_chunks` 占显存;
4. **协议是 algo 级属性**(`proto="Simple|LL|LL128"`),一个 XML 一种
   协议,不能按 pair 混用;
5. msccl-tools 的 `Check()` 只验 chunk 排列正确性,不验死锁。

### 8月8日(周六,半天)
设计文档定稿:MOSAIC 概念 → MSCCL 原语的映射表(见 9/10 归档版)。
关键决定:**epoch 在真机的载体 = "tb 内 step 顺序 + 双端一致的连接内
全序"**,因为 msccl 没有优先级抢占,顺序即优先级;wave 间依赖用
`depid`,但要做成"槽位级"而不是屏障(伏笔,8/26 会栽一次)。

---

## 第 2 周:planner 移植与正确性

### 8月10日
**改动**:新建 `gen/waterfill.py` + `gen/corridors.py` + 
`gen/emit_mscclang.py` 第一版(合计约 700 行):
- `waterfill.py::prior(demand, topo)`:逐 dst 列注水,输出每对 (s,d)
  在 {DIRECT, TX(via j), RX(via j'), SPREAD3} 上的字节劈分——算法
  照抄仿真 `scheduler/mosaic_schedule.h` 的注水语义,numpy 向量化;
- `corridors.py::assign_rails()`:corridor → 具体 rail(chan id)与
  中转卡编号;DIRECT 用本卡 rail,TX/RX 用邻卡 rail(邻卡选择 =
  (gpu_id + k) % 8 轮转,避免都压在 0 号卡);
- `emit_mscclang.py`:DSL 生成骨架:
  ```python
  topo = fully_connected(32)
  coll = AllToAll(32, chunks_per_pair_max, inplace=False)
  with MSCCLProgram("mosaic_a2a", topo, coll, instances=2,
                    protocol="Simple"):
      for (s, d, corridor, chunk_list) in plan:
          for ck in chunk_list:
              c = chunk(s, Buffer.input, ck.in_idx)
              if corridor.relay is None:            # DIRECT
                  c.copy(d, Buffer.output, ck.out_idx,
                         sendtb=ck.tb, recvtb=ck.rtb, ch=ck.chan)
              else:                                  # TX/RX:两段
                  m = c.copy(corridor.relay, Buffer.scratch, ck.slot,
                             sendtb=..., recvtb=..., ch=ck.nv_chan)
                  m.copy(d, Buffer.output, ck.out_idx,
                         sendtb=..., recvtb=..., ch=ck.chan)
      XML(); Check()
  ```
第一次 `XML()` python 进程 21GB OOM:512KB 粒度 → 4096 chunk/GPU,
DSL 中间图爆炸。**改动**:粒度提到 2MB(`gen/waves.py` 里
`GRANULE=2<<20`),7GB 能跑;emit 2.8s、XML 41MB——构造时间记账,
8/31 治理。

### 8月11日
uniform、DIRECT-only 先行(把 corridors 关掉最容易定位错):
`Check()` 过,真机 `verify()` fail——每对开头 2MB 对、后面错位。
**根因**:`Buffer.output` 的 index 单位是 chunk 且布局是"按 src 连续"
(NCCL alltoall 语义),我按"按 dst 连续"排的。
**改动**:`emit_mscclang.py::out_index()` 一个函数:
`out_idx = src * chunks_per_pair + k`(原来写成
`dst * chunks_per_pair + k`)。32/32 rank 校验全绿。
顺手在 bench 加 `--pattern-dump` 选项:错时打印首个错字节的
(expect, got) 解码出 (rank,peer,idx)——之后每次布局 bug 都靠它 5
分钟定位。

### 8月12日
开 TX/RX corridor。**坑:scratch 显存**。TX 每邻卡中转 ~8MB×31 对,
naive 排 `s_chunks=120`(×2MB×instances=2 ≈ 1GB/卡)。
**改动**:`gen/waves.py::alloc_scratch_slots()`——scratch 槽位按
wave 复用:wave i 的槽位集合与 wave i+1 完全重叠,复用点用
`depid` 把"wave i+1 的写"挂在"wave i 的读完成"上(DSL 里即后一个
copy 依赖前一个 copy 的 recv step)。`s_chunks` 120→24。
这就是仿真"wave = implicit flow control"的显存版:**wave 给了
scratch 一个 O(C/W) 的占用上界**,写进设计文档。

### 8月13日
协议实验。想给 ≤1MB 的冷对用 LL 降延迟——发现**做不到**:协议是
algo 级属性(8/7 清单第 4 条),一个 XML 只能一种。试着生成第二份
`proto="LL"` 的小消息专用 XML(两份一起加载,靠 nchunksperloop 不同
区分命中):真机校验 fail,只错尾块最后几个字节——LL 的 16B line
(8B 数据 + flag)对尾块有对齐要求,alltoallv 的变长尾块打不齐。
**决定**:放弃 LL,全场 `proto="Simple"`(62MB 量级本来就是吞吐主导,
LL128 又只保 NVLink 段)。删掉 LL 分支,`emit_mscclang.py` 里留一行
注释说明为什么不做。

### 8月14日
uniform 全 corridor 首个数字:**4.31ms**,比 NCCL(3.42)慢 26%。
profiling 三板斧第一次完整跑:
```bash
nsys profile -t cuda,nvtx,osrt -o mosaic_u1 mpirun ... ./bench_a2av
perfquery -x <lid> 1   # 每 NIC 口 PortXmitData,前后差分
top -H -p <pid>        # 看 ncclProxyProgress 线程
```
诊断:kernel `MSCCLKernel_Simple` 只 1 个 channel 有活,proxy 单线程
100%,8 块 NIC 只有 mlx5_0 出流量。**根因**:我所有 tb 的 `ch=0`
(corridors.py 里 rail 分配写了还没接到 emit)。

### 8月15日(周六)
接线:**改动** `emit_mscclang.py` 把 `ck.chan` 真正传进 `ch=`,并在
bench 启动脚本固定 HCA 枚举序,保证 planner 的 rail 编号和 NCCL 的
netDev 编号一致:
```bash
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
-x NCCL_DEBUG_SUBSYS=GRAPH,INIT   # 日志里核对 channel->netDev 绑定行
```
另外 nsys 看到每迭代 kernel 前 ~180µs 空洞 = 我在循环里重复
`cudaMemcpyAsync` demand 元数据,挪出循环。

---

## 第 3 周:性能调优(uniform 反超)

### 8月17日
8 channel 生效后 uniform **2.94ms**(NCCL 的 0.86×,反超 14%)。
交叉验证注水先验:`dcgmi dmon -e 1011,1012`(NVLink TX/RX 计数)算出
实际 TX:RX:DIRECT ≈ 21:21:58,planner 输出 22:22:56——**先验在真机
成立,误差 <2pt**。这张对照表后面答辩必用。

### 8月18日
instances 扫描(`--instances {1,2,4}` 重新 emit 三份 XML):
2 最好(2.87ms),4 掉到 3.3ms(tb 数翻倍后每 tb 数据太碎,proxy
FIFO 元数据开销占比上升;nsys 的 kernel 时长没涨,涨的是尾部等待)。
**定格 instances=2、8 chan**。
再开 PC:**改动** `gen/epoch_order.py` 第一版——tb 内 step 按
`(d−s) mod 32` 子序重排(只动发送端,伏笔!)。uniform 2.87→**2.79ms**
(−2.8%;仿真里 −9.5%,真机 FIFO 串行化天然吃掉一部分错峰收益,
合理,记入差异说明)。

### 8月19日
均值好了但 **p95=3.6ms** 抖。用 NCCL 的 proxy trace:
```bash
-x NCCL_PROXY_PROFILE=/tmp/proxy_%h_%p.json
python scripts/proxy_align.py /tmp/proxy_*.json   # 自写:按迭代对齐找慢连接
```
慢迭代全卡在 (rank3→17, rank11→25) 两条连接的 SendWait——这两对的
rail 恰好都落 leaf1→spine0 同一条上行。**真·ECMP hash collision**
(仿真 37 组 P3 的真机版)。
**处置**:`-x NCCL_IB_QPS_PER_CONNECTION=4`(多 QP 散 hash),p95
3.6→3.05ms;同时给网管提 spine adaptive-routing 工单。

### 8月20日
硬件排除法补齐(以后甩锅有据):`ib_write_bw` 全对分 25.1GB/s、
`p2pBandwidthLatencyTest` 双向 516GB/s,均达标。开始搬 zipf。

---

## 第 4 周:alltoallv(zipf)与死锁大战

### 8月24日
**改动**:`gen/waves.py` 支持变长——demand 按 512KB 粒度切变长
chunk 数(热对 18.3MB→37 chunk,冷对 1 chunk);`emit_mscclang.py`
里每 GPU 的 `i_chunks/o_chunks` 逐卡不同(XML 的 per-gpu 属性本来
就支持);`nchunksperloop` = 全局总 chunk 数,bench 的 count 配平。
`Check()` 过。真机第一跑:**hang**,30s 后
`[Rank 5] Timeout waiting for recv from 21`。

### 8月25日(死锁日,记详细)
复盘:PC 只重排了**发送端** step 序(8/18 的伏笔爆了);(peer,chan)
连接是 FIFO,rank21 的 recv 仍按旧序 post,rank5 的第 3 个 send 对上
第 1 个 recv 槽位,双方互等。仿真没这坑(min-epoch 是偏好不是闸),
**MSCCL 的 FIFO 是硬闸**。
候选修法:A. 收端同步重排(不够,跨 tb 的 recv 也要一致);
B. 每对独占 channel(32 上限,塞不下);**C. 立不变量:连接内序 =
全局 epoch 键序,双端同源生成**。选 C。
**改动**(`gen/epoch_order.py` 重写,~120 行):
1. 定义全局键 `key(ck) = (wave, hop, (d−s) mod K, seq)`;
2. planner 先按 (src,dst,chan) 聚合出每条连接的 chunk 列表,**发收
   两端都按同一 key 排序**后再切给各自 tb 生成 step——序从"发端
   属性"变成"连接属性";
3. 新建 `gen/check_dag.py`:把所有 step 建图(tb 内顺序边 + depid 边
   + 连接 FIFO 配对边),networkx 找环,50ms 跑完,**进 CI**:每次
   emit XML 后必跑。
改完死锁消失,又撞新 assert:`steps per tb 267 > 256`(热列 tb 步数
爆)。**改动**:`emit_mscclang.py::merge_steps()`——同连接内 epoch
连续且 buffer 偏移相邻的 chunk 合并成一个大 `cnt` step(等价仿真的
chunk merge),267→141 步。

### 8月26日
zipf 首个数字:**24.9ms**(1.23× LB;grouped send/recv 28.6)。
**改动**:`emit_mscclang.py` 顺便输出每 wave 的 chunk 边界 json,
bench 里 `nvtxRangePush("wave3")` 打标 → nsys timeline 直接能看
wave 结构。发现 wave3→4 之间 1.9ms 空洞,只在热 dst(rank24)同节点
邻卡上——8/12 的 scratch 依赖挂得太保守:我把 wave i+1 的中转写挂在
**wave i 的最后一个 step**(屏障语义)。
**改动**:`gen/waves.py::alloc_scratch_slots()` 依赖细化到**槽位级**
——wave i+1 用槽位 j 的写只依赖 wave i 里读槽位 j 的那一个 step。
空洞消失:**22.9ms(1.13×)**。这正对应仿真的"epoch 不是墙钟闸",
第一版我把它做成闸了,细化后才对齐语义。

### 8月27日
**改动**:把 34 组 diag-CoV 门抄进 `gen/epoch_order.py::pc_gate()`
(zipf 列 skew 对角相消 → 门开;persrc 类 demand 自动关),并在
`waterfill.py` 给热 dst 的 RX corridor 份额 +8%(等价 λ 的一次冷启
迭代,来自仿真经验值)。**22.3ms(1.10× LB,vs send/recv 1.28×,
vs padded 1.87×)**。

### 8月28日
correctness 加固日(全部进 `scripts/ci.sh`):
3 种 seed × 50 迭代全 pattern 校验、边界 case(全 0 行/31MB 单对/
不整除尾块)、`compute-sanitizer --tool memcheck`(慢 40×,跑通无错)、
`check_dag.py` 全 XML 扫描。全绿。

---

## 第 5 周:λ 反馈落地(profile 库)+ incast 微基准

### 8月31日
构造时间 + λ 反馈一起治理。约束:XML 在 comm init 时解析
(`MSCCL_XML_FILES`),**不能热换**。方案(对齐 38 组"离线产物"
哲学):
1. **profile 库**:λ/劈分空间量化成 12 格(热列强度 × 热点数),
   离线 emit 12 份 XML,`MSCCL_XML_FILES=p0.xml:p1.xml:...:p11.xml`
   一次全加载;
2. **运行时选格**:msccl 原生按 (collective, 整除性) 选算法,没有
   "手动指定第 i 个"的 API。**打补丁**(`patches/0001-force-algo.patch`,
   ~60 行):
   - `src/include/msccl.h`:声明 `ncclResult_t mscclSetForcedAlgo(int idx)`
     + 一个进程级 `volatile int mscclForcedAlgo = -1`;
   - `src/enqueue.cc` 的算法选择函数里:`if (mscclForcedAlgo >= 0)`
     直接返回该句柄(仍校验整除性,不合法则报错而不是静默回退);
   - 导出符号,bench/torch 侧 `dlsym` 调用。选择开销 = 读一个 int;
3. **反馈信号**:kernel 内测不了分 corridor 时间,用**每 rail 的 NIC
   字节计数差分**(`perfquery` 太慢,改读
   `/sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data`,
   每迭代 <30µs)+ 整体耗时,推 corridor 失衡方向,查表选格——
   这就是 λ 反馈的真机形态:测量→选格,而不是测量→改权重;
4. **漂移重建**:demand 走出 12 格(planner 里 KL>0.3)→ 后台线程
   emit 新 XML + `ncclCommInitRank` 建备用 comm(0.9s),**双 comm
   乒乓**切换,训练不停;
5. emit 加速:merge_steps 后 step 数少 10×,字符串拼接换 lxml,
   2.8s→**78ms/份**。

### 9月1日
drift 实验(bench 里 `--drift-at 25 --drift-seed 44`):
不切格 22.3→24.1ms(+8%);计数器反馈切格 **22.7ms**(+1.8%);
硬漂移(α→2.0)触发重建,3 步内恢复。λ 闭环成立。

### 9月2日
incast 微基准(对齐 37 组;demand 直接用仓库 `demand_n32_incast1.txt`
换算 62MB 预算):31→1,LB=1.92GB÷25GB/s=**76.9ms**。
- grouped send/recv 突发:**84.2ms**,且同机背景 allreduce 被拖慢
  2.3×(IB 无丢包,病灶是 PFC 反压的 congestion spreading);
- MOSAIC wave(W=6):**78.1ms(1.016×)**,背景波动 <5%;
- 证据链:`port_xmit_wait` 计数,send/recv 版 4.1e9 vs wave 版
  2.7e8——**wave 的 C/W 上界在真机显形为 XmitWait 15× 差**。
**改动**:bench 加 `--background allreduce` 模式(另开 comm 循环跑
64MB allreduce),量化"邻居无损"。

### 9月3日
PACE 补票:kernel 内做不了 admission,用"冷对降 chan 份额 + 后置
wave"在 profile 生成期近似(`corridors.py` 加 `pace_approx=True`)。
zipf 22.3→22.2ms,噪声级 → 按 fail.md 精神记"真机近似无肉",
默认关,代码留门。

### 9月4日
尺度旁支:16 GPU(2 节点)/24 GPU(3 节点)重跑 uniform+zipf
(planner 只改 n,一键),相对基线比值稳定 1.15-1.21×,排除
"4 节点特调"质疑。

---

## 第 6 周:MoE 端到端 + 收尾

### 9月7日
接 Megatron-DeepSpeed。**坑**:torch 的 `all_to_all_single` 带
`input_split_sizes` 走 NCCL grouped send/recv 路径,**永远不会命中
msccl 的 alltoall 算法**。
**改动**(`torch_ext/`):
1. `mosaic_a2a.cpp`:pybind 扩展,拿 `ProcessGroupNCCL` 里的
   `ncclComm_t`(私有,经 `unsafeGetComm` 风格的 hack 取出,注释里
   写明版本脆弱性),调 msccl fork 提供的 `ncclAllToAll` 扩展 API +
   `mscclSetForcedAlgo`;
2. `repack.cu`:两个轻量 kernel,把 torch 的变长布局 ⇄ 我们的 512KB
   粒度定长布局(0.21ms,计入所有对比);
3. `sharded_moe.py` 里 monkeypatch `_AllToAll.apply` → 我们的 op,
   带环境开关 `MOSAIC_A2A=1` 随时回退。

### 9月8日
端到端:GPT-MoE 1.3B、16 专家 top-2、EP=32,每层两次 A2A(实测
58-66MB/GPU,zipf 样热度)。
- baseline:step **412ms**(A2A 占 118ms);
- MOSAIC:A2A 96ms,step **391ms(−5.1%)**;
- 开反馈跑 6h:平均 **389ms**,无 hang,在线 checksum(每 100 步
  抽验一层)无报警。

### 9月9日
消融表(zipf,50 次,mean/p95)——每行只开一个新机制,和日记的
改动一一对应:

| 配置 | ms | vs LB | 对应改动 |
|---|---|---|---|
| grouped send/recv(调优后) | 27.9 / 31.8 | 1.37× | 8/4 + 9/14 网格 |
| + corridor 静态劈分 | 24.8 / 28.9 | 1.22× | 8/10-8/12 |
| + wave(槽位级依赖) | 23.4 / 26.1 | 1.15× | 8/12 + 8/26 |
| + PC(双端键序) | 22.7 / 24.9 | 1.12× | 8/18 + 8/25 |
| + λ profile 反馈 | **22.3 / 24.4** | **1.10×** | 8/31 |
| pad-to-max(反面) | 41.6 / 44.0 | 2.05× | — |

### 9月10-11日
归档:`profiles/` 12 份 XML + 生成命令 json、`patches/` 两个补丁
(force-algo、bench 用的 comm 双缓冲示例)、"仿真→真机对应表"
(corridor→tb/chan;epoch→连接内双端键序;wave→槽位级 depid;
λ→计数器反馈+选格;构造→查表 0.3µs/漂移乒乓 0.9s;buffer 定律→
scratch=O(C/W)+XmitWait)。

---

## 面试官三轮拷问(自审,修到无问题)

**第一轮(9月14日)**
1. *"割界当分母是给自己贴金?"* → 补"协议地板":同 traffic matrix 的
   裸 `ib_write_bw` 编排下限 21.4ms,MOSAIC 22.3 距地板 **1.04×**;
   表格双分母并列。
2. *"NCCL 基线调优了吗?"* → 没调过,补 `run_grid.sh` 扫
   `NCCL_ALGO/PROTO/NTHREADS/MIN_NCHANNELS` 36 组合,send/recv 最好
   28.6→**27.9ms**;全文所有对比换成调优后基线(领先 1.28×→1.25×,
   如实下调)。
3. *"死锁修复有证明吗?"* → 附录补:双端同键全序 ⇒ 每条 FIFO 序列
   逐元素匹配;跨连接只剩槽位 depid,wave 编号单调 ⇒ DAG 无环 ⇒
   无死锁;`check_dag.py` 作为机械检查常驻 CI。

**第二轮(9月15日)**
4. *"为什么不用 MSCCL++?"* → 补讨论:MSCCL++ 的 proxy/semaphore 原语
   能做真在线 λ 和 PACE,但要改应用 API;本次目标是 NCCL 语义原位
   替换,经典 msccl 是正确的第一站;给了 PACE 在 MSCCL++ 上的落法
   草图,列 future work。
5. *"ECMP 工单后续?"* → 网管开 spine AR 试点后 QP=1 与 QP=4 等效
   (p95 2.96 vs 3.05);交付配置保留 QP=4(不依赖网管)。
6. *"6h 就叫稳定?"* → 追加 48h 浸泡;期间节点 OOM 误触发 12 次乒乓
   重建,加 300ms 防抖后干净;写入已知问题。

**第三轮(9月16日)**
7. *"repack 开销在其他专家数下?"* → 扫 8/32/64 专家:0.11-0.38ms,
   占比 <2%;64 专家 profile 库 18 格、XML 驻留 41MB,可接受。
8. *"只留一条真机结论?"* → "wave 的 O(C/W) 不变量在真机同时兑现为
   scratch 显存、XmitWait 15×、邻居无损三件事——调度层一个不变量,
   顶掉买 buffer 和调 PFC 水线两类运维。"
9. 全表出处/seed/迭代数/误差棒复查 → 齐。**无遗留问题。**

---

## 最终成绩单(50 次均值,4×8 A100 + HDR)

| 场景 | 最好基线(调优后) | MOSAIC-MSCCL | 提升 | 距协议地板 |
|---|---|---|---|---|
| uniform 62MB | NCCL alltoall 3.31ms | **2.79ms** | 1.19× | 1.42× |
| zipf1.0 alltoallv | send/recv 27.9ms | **22.3ms** | **1.25×** | 1.04× |
| incast 31→1 | send/recv 84.2ms(邻居 ×2.3 拖累) | **78.1ms** | 1.08× + 邻居无损 | 1.02× |
| MoE 1.3B step | 412ms | **389ms** | −5.6% | — |
| per-A2A 构造 | —(静态方案需重编译) | 查表 0.3µs / 漂移乒乓 0.9s | — | — |

## 附:照做清单(最短路径)

1. 容器 `nvcr.io/nvidia/pytorch:22.12`;编 msccl fork(8/5 命令);
2. 跑 `alltoall_allpairs.py` 验证 MSCCL 加载(必须看到 Parsed 行;
   count 整除 nchunksperloop!);
3. 写 bench(8/4 结构:pattern 校验 + grouped send/recv 基线 + 双分母 LB);
4. planner 五件套:注水 → corridor/rail → wave/槽位 scratch →
   **双端同键全序**(死锁不变量)→ emit+merge_steps;每次 emit 后跑
   `check_dag.py` 和 `Check()`;
5. 调优顺序:channel↔rail 映射(HCA 枚举序固定)→ instances 扫描 →
   PC → p95 抖动(proxy trace 找慢连接,QP 散列);
6. λ:12 格 profile 库 + force-algo 补丁 + NIC 计数器选格 + 乒乓重建;
7. 接 torch:绕开 `all_to_all_single` 的 send/recv 路径,repack +
   `ncclAllToAll`;
8. 验收:消融表逐机制、双分母、48h 浸泡、面试官三问自审。

迁移完成。2026-09-16。
