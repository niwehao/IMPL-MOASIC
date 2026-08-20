// GPU 消费者内核(源文件,不进 Makefile)。
// PTX 由 tools/gen_consumer_ptx.sh 生成并嵌入 src/consumer_kernel_ptx.h,
// 运行时经 cuModuleLoadDataEx JIT(与 engine.cc 的自旋门核同一机制,免 nvcc)。
//
// **生命期铁律(坑 #36)**:核**每 collective 起停**,绝不常驻 ——
// `cudaFree` 做设备级隐式同步,会等所有在跑的 kernel;常驻核永不退出 ⇒
// torch 分配器一 free 就永久死锁(test_persist_sync 实测:常驻核在场时
// 十个 host API 里唯独 cudaFree 挂死)。有界生命期下 cudaFree 最多等一个
// collective 的核退出延迟(µs 级)。换代竞态由 **gen 代际号**根治:
// 核每轮比对 `*gen != my_gen` 立即退出,host 在启动新核前先 gen++ 杀老核。
//
// 契约(与 engine.cc / internal.h 的 KDesc 协议一致):
//   ring:pinned mapped 描述符环,每项 32B = {src, dst, len, pad}
//   ctr :pinned mapped 计数块,+0 head(host 写,已推入总数)
//                              +64 done(设备写,已完成总数,只增)
//                              +128 stop(host 写,~0=进行中;=N 表示收尾,
//                                        消费到 N 即退出)
//                              +160 input 就绪标志(host 回调写,engine 读)
//                              +192 gen(host 写,代际号;核见变即退)
//   prog:设备内存 u64[gridDim],各 block 的消费进度(跨 kernel 保留:
//         上一个 kernel 退出时 prog[b]==stop==下一个 start,天然衔接;
//         首次启动前 host cudaMemset 清零,start 亦为 0)
//   start:本次启动时的累计消费位置(== host 的 kdesc_pushed 快照)
//   cap :环容量(2 的幂)
//
// 完成判定全在设备:host 只写 pinned 内存(推描述符/写 stop),完成路径
// 零 CUDA 提交 —— 一期"设备等 host 标志 × host 提交拷贝"的死锁环不成立。
// 多 block 条带并行拷贝同一描述符;done 只能报**连续前缀**:块间快慢不一,
// 用 prog[] 的 min 汇报(plain store 竞态只会暂时少报,永不虚报 ——
// 任一时刻算出的 min 都是真完成前缀;不用原子,PCIe 映射内存无原子之忧)。

typedef unsigned long long u64;
typedef unsigned int u32;

// 输入就绪标志:排在 caller 流上(input 事件点之后),流执行到此 = a2a 输入
// 张量已产出。engine 纯读 pinned 标志判 hop0 可发 —— 异步窗口内 event 线程
// 一次 cudaEventQuery 都不能有(app 线程 torch sync 持锁时查询会冻死,实测)。
extern "C" __global__ void mosaic_flagset(u64 addr, u64 val) {
  *(volatile u64*)addr = val;
  __threadfence_system();
}

// 并行度模型(v2):block 按「描述符序号 % nch」分成 nch 个 channel,每
// channel BPG=gridDim/nch 个 block 条带同拷一个描述符 —— 多个描述符(不同
// peer 的发送/交付)并行搬运,而不是全体 block 串行逐个吃。done 仍 =
// min(prog[*]) 的连续前缀:非本 channel 的序号在 head 越过后立即计过,
// 本 channel 的序号拷完才计过 ⇒ 前缀语义不变。gridDim 必须整除 nch。
extern "C" __global__ void mosaic_consume(u64 ring_a, u64 ctr_a, u64 prog_a,
                                          u64 start, u32 cap, u64 my_gen, u32 nch) {
  volatile u64* head = (volatile u64*)(ctr_a);
  volatile u64* done = (volatile u64*)(ctr_a + 64);
  volatile u64* stop = (volatile u64*)(ctr_a + 128);
  volatile u64* inflag = (volatile u64*)(ctr_a + 160);
  volatile u64* gen = (volatile u64*)(ctr_a + 192);
  volatile u64* prog = (volatile u64*)prog_a;
  __shared__ u64 s_desc[3];
  __shared__ int s_state;
  // 输入门(设备侧):等 caller 流上的 flagset 核把 inflag 置 1 = 生产者
  // kernel 已写完 sendbuf、且 caller 流先前工作(含 caching allocator 复用块
  // 上的旧 kernel)全部完成。host 侧因此可以**立即**推描述符(消灭 host 观察
  // input 事件的往返延迟);本核不放行任何拷贝,读 sendbuf / 写 recvbuf 的
  // 顺序都由这道门保证。stop/gen 变化时照常退出(teardown 安全)。
  if (threadIdx.x == 0) {
    while (*inflag == 0 && *stop == ~0ull && *gen == my_gen) __nanosleep(2000);
  }
  __syncthreads();
  // done 刷新:min(prog) 只可能增长,但两个 block 并发"读-算-写"会让慢者的
  // 过期 min 覆盖新值;若之后全员进入等待就没人再写,done 永久卡低(实测
  // push=68 done=67 悬死)。因此:①只在 m 更大时写;②等待循环里持续刷新,
  // 过期覆盖必被下一次刷新纠正。
  auto refresh_done = [&](u64 m_floor) {
    u64 m = ~0ull;
    for (u32 b = 0; b < gridDim.x; b++) { u64 v = prog[b]; if (v < m) m = v; }
    if (m != ~0ull && m > m_floor && m > *done) *done = m;
  };
  const u32 bpg = gridDim.x / nch;          // 每 channel block 数
  const u32 mych = blockIdx.x / bpg;        // 本 block 的 channel
  const u32 lane = blockIdx.x % bpg;        // 组内条带角色
  // PCIe 轮询解耦:**每 channel 只有 lane0 轮询 pinned(head/stop/gen)**,
  // 并把进度广播到设备内存 chgo[ch](prog 区 +64 之后);组员只轮询设备字 ——
  // 轮询者数量 ∝ nch 而不是 blocks,拷贝宽度(bpg)可以放大而不打爆 PCIe。
  // 编码:低 63 位 = 可消费上界(head 或 stop),MSB = 退出信号。
  volatile u64* chgo = prog + 64 + mych;
  u64 i = start;
  for (;;) {
    if (threadIdx.x == 0) {
      if (lane != 0) {                      // 组员:等 lane0 的设备广播
        for (;;) {
          u64 g = *chgo;
          if (g & 0x8000000000000000ull) {
            u64 lim = g & ~0x8000000000000000ull;
            if (i >= lim) { s_state = 1; break; }
          }
          if ((g & ~0x8000000000000000ull) > i) { s_state = 0; break; }
          __nanosleep(500);
        }
        if (s_state == 0) {
          if ((u32)(i % nch) != mych) {
            s_state = 2;
          } else {
            __threadfence();
            volatile u64* d = (volatile u64*)(ring_a + (u64)((u32)i & (cap - 1)) * 32);
            s_desc[0] = d[0]; s_desc[1] = d[1]; s_desc[2] = d[2];
          }
        }
      } else {
      // **PCIe 纪律**:等待循环里的每一次 mapped-host 读写都是一笔 PCIe 事务。
      // 节拍取 1µs(500ns × 密集读会把 BAR 打满,饿死同一条 PCIe 上的 NIC
      // 门铃/镜像;4µs 则给每描述符交接叠 ~4µs 量化延迟)。
      // done 只由 block0 每 16 轮兜底刷一次。
      for (u32 spin = 0;; spin++) {
        if (*head > i) { s_state = 0; break; }
        u64 sp = *stop;
        if (sp != ~0ull && i >= sp) { s_state = 1; break; }
        if (*gen != my_gen) { s_state = 1; break; }   // 换代:立即让位(坑 #36)
        if (blockIdx.x == 0 && (spin & 15) == 0) refresh_done(start);
        __nanosleep(1000);
      }
      if (s_state == 0) {
        *chgo = i + 1;                        // 广播:本 channel 可消费到 i+1
        if ((u32)(i % nch) != mych) {
          s_state = 2;                        // 别的 channel 的描述符:计过不拷
        } else {
          __threadfence_system();             // head 之后再读描述符(acquire)
          volatile u64* d = (volatile u64*)(ring_a + (u64)((u32)i & (cap - 1)) * 32);
          s_desc[0] = d[0]; s_desc[1] = d[1]; s_desc[2] = d[2];
        }
      } else {
        *chgo = i | 0x8000000000000000ull;    // 退出广播(上界 = i,组员干完即退)
      }
      }
    }
    __syncthreads();
    if (s_state == 1) break;
    if (s_state == 2) {                       // 跳过:推进 prog 即可
      __syncthreads();
      if (threadIdx.x == 0) {
        prog[blockIdx.x] = i + 1;
        __threadfence();
        __threadfence_system();
        if ((i & 15) == 0) refresh_done(start);
      }
      i++;
      __syncthreads();
      continue;
    }
    const u64 len = s_desc[2];
    // 条带划分(channel 组内 bpg 个 block 分同一描述符):chunk 16B 对齐
    const u64 chunk = ((len + bpg - 1) / bpg + 15) & ~15ull;
    const u64 lo = (u64)lane * chunk;
    const u64 hi = lo + chunk < len ? lo + chunk : len;
    if (lo < len) {
      const unsigned char* src = (const unsigned char*)s_desc[0] + lo;
      unsigned char* dst = (unsigned char*)s_desc[1] + lo;
      const u64 n = hi - lo;
      if ((((u64)src | (u64)dst) & 15) == 0) {           // 16B 向量拷贝
        const u64 n16 = n >> 4;
        const ulonglong2* s2 = (const ulonglong2*)src;
        ulonglong2* d2 = (ulonglong2*)dst;
        for (u64 k = threadIdx.x; k < n16; k += blockDim.x) d2[k] = s2[k];
        for (u64 k = (n16 << 4) + threadIdx.x; k < n; k += blockDim.x)
          dst[k] = src[k];
      } else if ((((u64)src | (u64)dst) & 7) == 0) {
        const u64 n8 = n >> 3;
        for (u64 k = threadIdx.x; k < n8; k += blockDim.x)
          ((u64*)dst)[k] = ((const u64*)src)[k];
        for (u64 k = (n8 << 3) + threadIdx.x; k < n; k += blockDim.x)
          dst[k] = src[k];
      } else {
        for (u64 k = threadIdx.x; k < n; k += blockDim.x)
          dst[k] = src[k];
      }
    }
    __threadfence_system();                   // 本 block 的 payload 写先于 prog
    __syncthreads();
    if (threadIdx.x == 0) {
      prog[blockIdx.x] = i + 1;
      __threadfence();                        // prog 对其他 block 可见
      __threadfence_system();
      refresh_done(start);
    }
    i++;
    __syncthreads();                          // s_desc/s_state 复用防竞态
  }
  if (threadIdx.x == 0) refresh_done(start);  // 退出前最后一刷(等待方兜底)
}
