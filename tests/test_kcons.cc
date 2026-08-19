// S3 二期消费者内核独立单测(无 torch/无网络,单 GPU):
//   1) JIT 加载 consumer_kernel_ptx.h;
//   2) 分配描述符环/计数块(pinned mapped)+ prog(设备);
//   3) 启动核,分批推入 D2D 拷贝描述符(乱长度/乱对齐),轮询 done;
//   4) 写 stop 收尾,校验核退出(stream 变空)与所有字节逐位正确;
//   5) 第二轮再启动(prog 跨核衔接语义)。
// 过 = 输出 "KCONS PASS";任何失配/超时 = 非零退出。
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>

#include "../src/consumer_kernel_ptx.h"

typedef int (*cu_mload_fn)(void**, const void*, unsigned int, void*, void**);
typedef int (*cu_mfunc_fn)(void**, void*, const char*);
typedef int (*cu_launch_fn)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, void*, void**, void**);

static const unsigned BLOCKS = 4, THREADS = 256;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct KDesc { uint64_t src, dst, len, pad; };

int main() {
  CK(cudaSetDevice(0));
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "no libcuda\n"); return 1; }
  auto mload = (cu_mload_fn)dlsym(h, "cuModuleLoadDataEx");
  auto mfunc = (cu_mfunc_fn)dlsym(h, "cuModuleGetFunction");
  auto launch = (cu_launch_fn)dlsym(h, "cuLaunchKernel");
  if (!mload || !mfunc || !launch) { fprintf(stderr, "no drv syms\n"); return 1; }
  // 先触发 runtime 建 context(driver API 依赖)
  CK(cudaFree(0));
  void* mod = nullptr;
  if (mload(&mod, kConsumerPtx, 0, nullptr, nullptr) != 0) { fprintf(stderr, "JIT fail\n"); return 1; }
  void* fn = nullptr;
  if (mfunc(&fn, mod, "mosaic_consume") != 0) { fprintf(stderr, "no entry\n"); return 1; }

  const uint32_t CAP = 256;
  KDesc* ring = nullptr; uint8_t* ctr = nullptr;
  CK(cudaHostAlloc((void**)&ring, CAP * sizeof(KDesc), cudaHostAllocMapped | cudaHostAllocPortable));
  CK(cudaHostAlloc((void**)&ctr, 256, cudaHostAllocMapped | cudaHostAllocPortable));
  void* ring_d = nullptr; void* ctr_d = nullptr; void* prog_d = nullptr;
  CK(cudaHostGetDevicePointer(&ring_d, ring, 0));
  CK(cudaHostGetDevicePointer(&ctr_d, ctr, 0));
  CK(cudaMalloc(&prog_d, BLOCKS * sizeof(uint64_t)));
  CK(cudaMemset(prog_d, 0, BLOCKS * sizeof(uint64_t)));
  memset(ctr, 0, 256);
  *(uint64_t*)(ctr + 128) = ~0ull;
  *(uint64_t*)(ctr + 160) = 1;   // 输入门:单测输入常备,直接开
  *(uint64_t*)(ctr + 192) = 1;   // gen(坑 #36 代际号;单测固定一代)

  cudaStream_t sg;
  CK(cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking));

  const size_t POOL = 8u << 20;
  uint8_t* src_d = nullptr; uint8_t* dst_d = nullptr;
  CK(cudaMalloc(&src_d, POOL)); CK(cudaMalloc(&dst_d, POOL));
  std::vector<uint8_t> host_src(POOL), host_chk(POOL);
  srand(20260816);
  for (size_t i = 0; i < POOL; i++) host_src[i] = (uint8_t)rand();
  CK(cudaMemcpy(src_d, host_src.data(), POOL, cudaMemcpyHostToDevice));

  uint64_t pushed = 0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  for (int round = 0; round < 2; round++) {
    CK(cudaMemset(dst_d, 0, POOL));
    *(uint64_t*)(ctr + 128) = ~0ull;                     // stop 复位
    __sync_synchronize();
    unsigned long long ring_a = (unsigned long long)(uintptr_t)ring_d,
                       ctr_a = (unsigned long long)(uintptr_t)ctr_d,
                       prog_a = (unsigned long long)(uintptr_t)prog_d,
                       start = pushed, my_gen = 1;
    unsigned int cap = CAP;
    unsigned int nch = 2;   // 单测覆盖多 channel 语义(grid=4 → bpg=2)
    void* params[7] = {&ring_a, &ctr_a, &prog_a, &start, &cap, &my_gen, &nch};
    if (launch(fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, sg, params, nullptr) != 0) {
      fprintf(stderr, "launch fail\n"); return 1;
    }
    // 乱长度/乱对齐描述符(覆盖 8B 对齐路径与逐字节路径、条带边界)
    struct Seg { size_t so, do_, ln; };
    std::vector<Seg> segs;
    size_t so = 0, dof = 0;
    const size_t lens[] = {1, 7, 8, 63, 4096, 1 << 20, 12345, 65536, 3, 800000};
    for (int rep = 0; rep < 40; rep++) {
      size_t ln = lens[rep % 10] + (rep % 3);
      size_t mis = rep % 5;                              // 制造错位
      if (so + mis + ln > POOL || dof + mis + ln > POOL) break;
      segs.push_back({so + mis, dof + mis, ln});
      so += mis + ln; dof += mis + ln;
    }
    size_t next = 0;
    auto t0 = now();
    while (true) {
      uint64_t done = __atomic_load_n((uint64_t*)(ctr + 64), __ATOMIC_ACQUIRE);
      // 分批推(制造核追赶/等待两种相位)
      while (next < segs.size() && pushed - done < CAP) {
        KDesc& d = ring[pushed & (CAP - 1)];
        d.src = (uint64_t)(uintptr_t)(src_d + segs[next].so);
        d.dst = (uint64_t)(uintptr_t)(dst_d + segs[next].do_);
        d.len = segs[next].ln;
        __atomic_store_n((uint64_t*)(ctr + 0), pushed + 1, __ATOMIC_RELEASE);
        pushed++; next++;
        if (next % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
      if (next == segs.size() && done == pushed) break;
      if (now() - t0 > std::chrono::seconds(20)) { fprintf(stderr, "TIMEOUT done=%llu pushed=%llu\n",
        (unsigned long long)done, (unsigned long long)pushed); return 1; }
    }
    // 收尾:写 stop,核应退出 → stream 变空
    __atomic_store_n((uint64_t*)(ctr + 128), pushed, __ATOMIC_RELEASE);
    t0 = now();
    while (cudaStreamQuery(sg) != cudaSuccess) {
      if (now() - t0 > std::chrono::seconds(10)) { fprintf(stderr, "kernel 未退出\n"); return 1; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // 校验
    CK(cudaMemcpy(host_chk.data(), dst_d, POOL, cudaMemcpyDeviceToHost));
    for (auto& s : segs)
      if (memcmp(host_chk.data() + s.do_, host_src.data() + s.so, s.ln) != 0) {
        fprintf(stderr, "MISMATCH so=%zu do=%zu ln=%zu (round %d)\n", s.so, s.do_, s.ln, round);
        return 1;
      }
    printf("round %d ok:%zu 段,%llu 描述符累计\n", round, segs.size(),
           (unsigned long long)pushed);
  }
  // ── 代际退出(坑 #36 的命脉):核在等描述符时,host 把 gen 一改就必须退出。
  // 核绝不允许常驻 —— 常驻 × torch 的 cudaFree(设备级隐式同步)= 永久死锁。
  {
    *(uint64_t*)(ctr + 128) = ~0ull;                     // stop 复位(进行中)
    __sync_synchronize();
    unsigned long long ring_a = (unsigned long long)(uintptr_t)ring_d,
                       ctr_a = (unsigned long long)(uintptr_t)ctr_d,
                       prog_a = (unsigned long long)(uintptr_t)prog_d,
                       start = pushed, my_gen = 1;
    unsigned int cap = CAP;
    unsigned int nch = 2;   // 单测覆盖多 channel 语义(grid=4 → bpg=2)
    void* params[7] = {&ring_a, &ctr_a, &prog_a, &start, &cap, &my_gen, &nch};
    if (launch(fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, sg, params, nullptr) != 0) {
      fprintf(stderr, "gen-test launch fail\n"); return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (cudaStreamQuery(sg) == cudaSuccess) { fprintf(stderr, "核没在等就退了\n"); return 1; }
    __atomic_store_n((uint64_t*)(ctr + 192), 2ull, __ATOMIC_RELEASE);   // 换代
    auto t0 = now();
    while (cudaStreamQuery(sg) != cudaSuccess) {
      if (now() - t0 > std::chrono::seconds(5)) {
        fprintf(stderr, "GEN 悬死:gen 改了核仍不退出(常驻 = cudaFree 死锁,坑 #36)\n");
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    printf("gen ok:换代后 %.1fms 内退出\n",
           std::chrono::duration<double, std::milli>(now() - t0).count());
    *(uint64_t*)(ctr + 192) = 1;                          // 复位给后续用
  }

  // ── 自旋门核单测(mosaic_gate:等 *addr >= want;host 纯 pinned 写开门)
  {
    static const char* kGatePtx =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry mosaic_gate(.param .u64 p_addr, .param .u32 p_want){\n"
        " .reg .pred %p1;\n .reg .b32 %r1,%r2;\n .reg .b64 %rd1;\n"
        " ld.param.u64 %rd1,[p_addr];\n ld.param.u32 %r2,[p_want];\n"
        "$L_loop:\n ld.volatile.global.u32 %r1,[%rd1];\n"
        " setp.lt.u32 %p1,%r1,%r2;\n @%p1 nanosleep.u32 500;\n @%p1 bra $L_loop;\n ret;\n}\n";
    void* gmod = nullptr; void* gfn = nullptr;
    if (mload(&gmod, kGatePtx, 0, nullptr, nullptr) != 0 ||
        mfunc(&gfn, gmod, "mosaic_gate") != 0) { fprintf(stderr, "gate JIT fail\n"); return 1; }
    uint32_t* flag = nullptr;
    CK(cudaHostAlloc((void**)&flag, 4, cudaHostAllocMapped | cudaHostAllocPortable));
    *flag = 0;
    void* flag_d = nullptr;
    CK(cudaHostGetDevicePointer(&flag_d, flag, 0));
    cudaStream_t gs;
    CK(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking));
    unsigned long long addr = (unsigned long long)(uintptr_t)flag_d;
    unsigned int want = 1;
    void* gp[2] = {&addr, &want};
    if (launch(gfn, 1, 1, 1, 1, 1, 1, 0, gs, gp, nullptr) != 0) {
      fprintf(stderr, "gate launch fail\n"); return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (cudaStreamQuery(gs) == cudaSuccess) { fprintf(stderr, "gate 未等待就退出\n"); return 1; }
    __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);   // 开门
    auto t0 = now();
    while (cudaStreamQuery(gs) != cudaSuccess) {
      if (now() - t0 > std::chrono::seconds(5)) { fprintf(stderr, "GATE 悬死:host 写 flag 后核不退出\n"); return 1; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    printf("gate ok:flag 写后 %.1fms 内开门\n",
           std::chrono::duration<double, std::milli>(now() - t0).count());

    // ── 实战形态复现:两个常驻自旋核(消费者/镜像的替身,各占一条硬件通道)
    // 在场时,门核放 **legacy 流(0x1)** 还能不能开(MoE 现场 132214 的问题)
    uint64_t* stop2 = nullptr;
    CK(cudaHostAlloc((void**)&stop2, 8, cudaHostAllocMapped | cudaHostAllocPortable));
    *stop2 = 0;
    void* stop2_d = nullptr;
    CK(cudaHostGetDevicePointer(&stop2_d, stop2, 0));
    cudaStream_t r1s, r2s;
    CK(cudaStreamCreateWithFlags(&r1s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&r2s, cudaStreamNonBlocking));
    unsigned long long saddr = (unsigned long long)(uintptr_t)stop2_d;
    unsigned int sw = 1;
    void* sp[2] = {&saddr, &sw};
    // 常驻替身 = 同一个 gate 核等一个不会很快满足的 flag
    if (launch(gfn, 1,1,1, 1,1,1, 0, r1s, sp, nullptr) != 0 ||
        launch(gfn, 1,1,1, 1,1,1, 0, r2s, sp, nullptr) != 0) {
      fprintf(stderr, "resident launch fail\n"); return 1;
    }
    *flag = 0;
    __sync_synchronize();
    unsigned int want2 = 2;
    void* gp2[2] = {&addr, &want2};
    if (launch(gfn, 1,1,1, 1,1,1, 0, (void*)0x1 /*CU_STREAM_LEGACY*/, gp2, nullptr) != 0) {
      fprintf(stderr, "legacy gate launch fail\n"); return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    __atomic_store_n(flag, 2u, __ATOMIC_RELEASE);
    t0 = now();
    bool opened = false;
    while (now() - t0 < std::chrono::seconds(5)) {
      cudaError_t q = cudaStreamQuery((cudaStream_t)0x1);
      if (q == cudaSuccess) { opened = true; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    __atomic_store_n(stop2, 1ull, __ATOMIC_RELEASE);   // 放常驻替身
    cudaStreamSynchronize(r1s); cudaStreamSynchronize(r2s);
    if (!opened) { fprintf(stderr, "GATE-LEGACY 悬死:常驻核在场时 legacy 门不开\n"); return 1; }
    printf("gate-legacy ok:常驻×2 在场,legacy 门 %.1fms 开\n",
           std::chrono::duration<double, std::milli>(now() - t0).count());
  }
  printf("KCONS PASS\n");
  return 0;
}
