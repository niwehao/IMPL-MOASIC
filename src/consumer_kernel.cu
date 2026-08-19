// S3 二期方案 N:GPU 消费者内核(源文件,不进 Makefile)。
// PTX 由 tools/gen_consumer_ptx.sh 生成并嵌入 src/consumer_kernel_ptx.h,
// 运行时经 cuModuleLoadDataEx JIT(与 engine.cc 的自旋门核同一机制,免 nvcc)。
//
// **生命期铁律(坑 #36)**:两个核都是**每 collective 起停**,绝不常驻 ——
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

// ── 镜像核配置(pinned mapped;host 在 setup 填好,launch 只传一个指针)
// 异步窗口内 s_poll 的 cudaMemcpyAsync 镜像不可用(门等待未决期间流上操作
// 被驱动楔死,Raven 实测),由本核设备直读 sig/credit/ring 写 shadow。
struct MirrorCfg {
  u64 sig, credit, ring;                      // 设备(对称堆)
  u64 sig_shadow, hdr_shadow, credit_shadow;  // pinned mapped(设备侧地址)
  u64 last_sig;                               // 设备 scratch [n*nslots]
  u64 stop;                                   // 与消费者核同一停止字(ctr+128)
  u64 sig_raw;                                // 诊断:无条件镜像的原始 sig
  u32 n, nslots, slot_bytes, hdr_stride, hdr_bytes;
  u32 magic_val;                              // SliceHdr.magic 期望值
  u32 tag_check;                              // 1=校验 hdr.crc==槽使用代数(sig 值)
  u32 pad;
};

// 输入就绪标志:排在 caller 流上(input 事件点之后),流执行到此 = a2a 输入
// 张量已产出。engine 纯读 pinned 标志判 hop0 可发 —— 异步窗口内 event 线程
// 一次 cudaEventQuery 都不能有(app 线程 torch sync 持锁时查询会冻死,实测)。
extern "C" __global__ void mosaic_flagset(u64 addr, u64 val) {
  *(volatile u64*)addr = val;
  __threadfence_system();
}

extern "C" __global__ void mosaic_mirror(u64 cfg_a, u64 my_gen) {
  const MirrorCfg* g = (const MirrorCfg*)cfg_a;
  volatile u64* stop = (volatile u64*)g->stop;
  volatile u64* gen = (volatile u64*)(g->stop + 64);   // ctr+192(stop 在 +128)
  volatile u64* sig = (volatile u64*)g->sig;
  volatile u64* ssh = (volatile u64*)g->sig_shadow;
  volatile u64* cr = (volatile u64*)g->credit;
  volatile u64* csh = (volatile u64*)g->credit_shadow;
  u64* last = (u64*)g->last_sig;
  volatile u64* raw = (volatile u64*)g->sig_raw;
  const u32 total = g->n * g->nslots;
  while (*stop == ~0ull && *gen == my_gen) {   // 收尾或换代都退(坑 #36)
    for (u32 i = threadIdx.x; i < g->n; i += blockDim.x) csh[i] = cr[i];
    for (u32 i = threadIdx.x; i < total; i += blockDim.x) {
      u64 v = sig[i];
      if (v != last[i]) {
        if (raw) raw[i] = v;                  // 诊断:未校验的原始 sig(仅变化时写)
        // 槽头(64B)先落 shadow,并校验 magic(+ 使用代数 tag,置于
        // SliceHdr.crc 偏移 30,host-API 发送时写入 = sig 值):sig 与
        // hdr/payload 若越序/丢失,头没就绪或还是上代的 —— 不发布、不记
        // last,下轮重拷直到一致(120556 实测:一片 sig 先到旧头,按序
        // 消费被永久卡死)。通过校验才 fence + 发布 sig。
        const unsigned char* src = (const unsigned char*)g->ring + (u64)i * g->slot_bytes;
        unsigned char* dst = (unsigned char*)g->hdr_shadow + (u64)i * g->hdr_stride;
        u32 magic = 0, tag = 0;
        for (u32 b = 0; b < g->hdr_bytes; b++) {
          unsigned char x = src[b];
          dst[b] = x;
          if (b < 4) magic |= (u32)x << (8 * b);
          if (b >= 30 && b < 34) tag |= (u32)x << (8 * (b - 30));
        }
        if (magic == g->magic_val && (!g->tag_check || tag == (u32)v)) {
          __threadfence_system();
          ssh[i] = v;
          last[i] = v;
        }
      }
    }
    __nanosleep(20000);                       // ~20µs 镜像节拍
  }
}

extern "C" __global__ void mosaic_consume(u64 ring_a, u64 ctr_a, u64 prog_a,
                                          u64 start, u32 cap, u64 my_gen) {
  volatile u64* head = (volatile u64*)(ctr_a);
  volatile u64* done = (volatile u64*)(ctr_a + 64);
  volatile u64* stop = (volatile u64*)(ctr_a + 128);
  volatile u64* gen = (volatile u64*)(ctr_a + 192);
  volatile u64* prog = (volatile u64*)prog_a;
  __shared__ u64 s_desc[3];
  __shared__ int s_state;
  // done 刷新:min(prog) 只可能增长,但两个 block 并发"读-算-写"会让慢者的
  // 过期 min 覆盖新值;若之后全员进入等待就没人再写,done 永久卡低(实测
  // push=68 done=67 悬死)。因此:①只在 m 更大时写;②等待循环里持续刷新,
  // 过期覆盖必被下一次刷新纠正。
  auto refresh_done = [&](u64 m_floor) {
    u64 m = ~0ull;
    for (u32 b = 0; b < gridDim.x; b++) { u64 v = prog[b]; if (v < m) m = v; }
    if (m != ~0ull && m > m_floor && m > *done) *done = m;
  };
  u64 i = start;
  for (;;) {
    if (threadIdx.x == 0) {
      // **PCIe 纪律(坑 #38)**:等待循环里的每一次 mapped-host 读写都是
      // 一笔 PCIe 事务。旧版 4 个 block × 每 500ns 读 head/stop/gen + 刷 done,
      // 把 BAR 通道打满 —— 同一条 PCIe 上的 s_poll D2H 镜像与 NVSHMEM proxy
      // 门铃被饿死(现场:stx/evpoll 双 600、镜像轮转冻结、put 不发出)。
      // 现在:节拍 4µs、done 只由 block0 每 16 轮兜底刷一次。
      for (u32 spin = 0;; spin++) {
        if (*head > i) { s_state = 0; break; }
        u64 sp = *stop;
        if (sp != ~0ull && i >= sp) { s_state = 1; break; }
        if (*gen != my_gen) { s_state = 1; break; }   // 换代:立即让位(坑 #36)
        if (blockIdx.x == 0 && (spin & 15) == 0) refresh_done(start);
        __nanosleep(4000);
      }
      if (s_state == 0) {
        __threadfence_system();               // head 之后再读描述符(acquire)
        volatile u64* d = (volatile u64*)(ring_a + (u64)((u32)i & (cap - 1)) * 32);
        s_desc[0] = d[0]; s_desc[1] = d[1]; s_desc[2] = d[2];
      }
    }
    __syncthreads();
    if (s_state == 1) break;
    const u64 len = s_desc[2];
    // 条带划分:block j 拷 [j*chunk, min(len,(j+1)*chunk));chunk 8B 对齐,
    // 使各条带首地址相对 8B 网格不漂移
    const u64 chunk = ((len + gridDim.x - 1) / gridDim.x + 7) & ~7ull;
    const u64 lo = (u64)blockIdx.x * chunk;
    const u64 hi = lo + chunk < len ? lo + chunk : len;
    if (lo < len) {
      const unsigned char* src = (const unsigned char*)s_desc[0] + lo;
      unsigned char* dst = (unsigned char*)s_desc[1] + lo;
      const u64 n = hi - lo;
      if ((((u64)src | (u64)dst) & 7) == 0) {
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
