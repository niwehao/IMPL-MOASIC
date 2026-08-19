// 平台探针:**常驻(永不退出)核在场时,host 侧哪些 CUDA API 会被卡死**。
//
// 动机(坑 #35):S3 异步档下,两个 collective 引擎侧全部完成、门也开了,
// app 却恒卡在 torch 的 sort_chunks(torch.split/cat + tolist)。所有失败
// 配置的唯一共同点 = **常驻消费者核**(NOGATE 失败排除门、S3_STREAMS 失败
// 排除镜像/host-API 发送、默认档通过)。头号嫌疑:cudaFree 之类做**设备级
// 隐式同步**的 API 会等常驻核 —— 而它永不退出 ⇒ 永久死锁。坑 #28 里我自己
// 的 cudaMalloc 撞过同一堵墙,但 **torch 的分配我无法前置**。
//
// 本探针把嫌疑 API 逐个在"常驻核在场"下跑一遍(每个 3s 超时),报告第一个
// 挂住的;跑之前先做一遍无核对照。结论直接决定 S3 的架构走向:
//   - 若 cudaFree/cudaMalloc 挂 ⇒ 常驻核方案在 torch 进程里不可行,
//     必须改**有界生命期核**(每 collective 起停)或独立 context。
//   - 若全不挂 ⇒ 假设被推翻,回去查 torch 侧。
// 用法:./build/test_persist_sync   (单 GPU,无网络;不进 regression 判据)
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

typedef int (*cu_mload_fn)(void**, const void*, unsigned int, void*, void**);
typedef int (*cu_mfunc_fn)(void**, void*, const char*);
typedef int (*cu_launch_fn)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, void*, void**, void**);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// 与 engine.cc 的门核同源:while (*addr < want) nanosleep(500)
static const char* kGatePtx =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry mosaic_gate(.param .u64 p_addr, .param .u32 p_want){\n"
    " .reg .pred %p1;\n .reg .b32 %r1,%r2;\n .reg .b64 %rd1;\n"
    " ld.param.u64 %rd1,[p_addr];\n ld.param.u32 %r2,[p_want];\n"
    "$L_loop:\n ld.volatile.global.u32 %r1,[%rd1];\n"
    " setp.lt.u32 %p1,%r1,%r2;\n @%p1 nanosleep.u32 500;\n @%p1 bra $L_loop;\n ret;\n}\n";

struct Probe { const char* name; std::function<void()> fn; };

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);   // 崩溃也不丢日志
  fprintf(stderr, "[dbg] setDevice\n");
  CK(cudaSetDevice(0));
  CK(cudaFree(0));                       // 建 context
  fprintf(stderr, "[dbg] dlopen\n");
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "no libcuda\n"); return 1; }
  auto mload = (cu_mload_fn)dlsym(h, "cuModuleLoadDataEx");
  auto mfunc = (cu_mfunc_fn)dlsym(h, "cuModuleGetFunction");
  auto launch = (cu_launch_fn)dlsym(h, "cuLaunchKernel");
  fprintf(stderr, "[dbg] JIT\n");
  void* mod = nullptr; void* fn = nullptr;
  if (mload(&mod, kGatePtx, 0, nullptr, nullptr) != 0 ||
      mfunc(&fn, mod, "mosaic_gate") != 0) { fprintf(stderr, "JIT fail\n"); return 1; }

  // 探针要用的资源全部**先**备好(常驻核起来之后再分配就是在测自己)
  fprintf(stderr, "[dbg] alloc\n");
  uint32_t* flag = nullptr;
  CK(cudaHostAlloc((void**)&flag, 4, cudaHostAllocMapped | cudaHostAllocPortable));
  *flag = 0;
  void* flag_d = nullptr;
  CK(cudaHostGetDevicePointer(&flag_d, flag, 0));
  void* dbuf = nullptr;
  CK(cudaMalloc(&dbuf, 1 << 20));
  void* dbuf2 = nullptr;                 // 给 cudaFree 探针专用(先分配好)
  CK(cudaMalloc(&dbuf2, 1 << 20));
  std::vector<uint8_t> hbuf(4096);
  cudaStream_t s_other;
  CK(cudaStreamCreateWithFlags(&s_other, cudaStreamNonBlocking));
  cudaEvent_t ev;
  CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CK(cudaEventRecord(ev, s_other));
  cudaStream_t s_persist;
  CK(cudaStreamCreateWithFlags(&s_persist, cudaStreamNonBlocking));

  void* tmp_dev = nullptr;               // cudaMalloc 探针的产物,末尾统一释放
  void* tmp_host = nullptr;

  fprintf(stderr, "[dbg] probes 构建\n");
  std::vector<Probe> probes = {
      {"cudaLaunchKernel(其它流)", [&] {
         unsigned long long a = (unsigned long long)(uintptr_t)flag_d;
         unsigned int w = 0;             // want=0 ⇒ 立刻退出
         void* p[2] = {&a, &w};
         launch(fn, 1, 1, 1, 1, 1, 1, 0, s_other, p, nullptr);
       }},
      {"cudaEventQuery", [&] { cudaEventQuery(ev); }},
      {"cudaStreamQuery(其它流)", [&] { cudaStreamQuery(s_other); }},
      {"cudaStreamSynchronize(其它流)", [&] { cudaStreamSynchronize(s_other); }},
      {"cudaMemcpyAsync+Sync(其它流)", [&] {
         cudaMemcpyAsync(hbuf.data(), dbuf, 4096, cudaMemcpyDeviceToHost, s_other);
         cudaStreamSynchronize(s_other);
       }},
      {"cudaMemcpy(同步,legacy 流)", [&] {
         cudaMemcpy(hbuf.data(), dbuf, 4096, cudaMemcpyDeviceToHost);
       }},
      {"cudaMalloc", [&] { cudaMalloc(&tmp_dev, 1 << 20); }},
      {"cudaFree  ← 头号嫌疑", [&] { cudaFree(dbuf2); dbuf2 = nullptr; }},
      {"cudaHostAlloc", [&] { cudaHostAlloc(&tmp_host, 1 << 20, cudaHostAllocDefault); }},
      {"cudaFreeHost", [&] { if (tmp_host) { cudaFreeHost(tmp_host); tmp_host = nullptr; } }},
  };

  auto run_probes = [&](const char* phase, int tmo_ms) -> const char* {
    printf("== %s ==\n", phase);
    for (auto& p : probes) {
      std::atomic<bool> done{false};
      std::thread th([&] { p.fn(); done.store(true); });
      auto t0 = std::chrono::steady_clock::now();
      while (!done.load() &&
             std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(tmo_ms))
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (!done.load()) {
        printf("  **HANG** %s(>%dms)\n", p.name, tmo_ms);
        th.detach();                     // 由调用方放常驻核后自然解开
        return p.name;
      }
      th.join();
      printf("  ok   %s\n", p.name);
    }
    return nullptr;
  };

  // 对照组:无常驻核
  fprintf(stderr, "[dbg] 对照开始\n");
  if (run_probes("对照:无常驻核", 3000)) {
    fprintf(stderr, "对照组就挂了 —— 环境有别的问题,结论作废\n");
    return 1;
  }

  // 实验组:常驻核(等一个我们暂不设置的 flag)在场
  CK(cudaMalloc(&dbuf2, 1 << 20));       // 补回 cudaFree 探针的弹药
  {
    unsigned long long a = (unsigned long long)(uintptr_t)flag_d;
    unsigned int w = 1;
    void* p[2] = {&a, &w};
    if (launch(fn, 1, 1, 1, 1, 1, 1, 0, s_persist, p, nullptr) != 0) {
      fprintf(stderr, "persist launch fail\n"); return 1;
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  if (cudaStreamQuery(s_persist) == cudaSuccess) {
    fprintf(stderr, "常驻核没常驻(提前退出),探针无效\n"); return 1;
  }
  const char* hung = run_probes("实验:常驻核在场", 3000);

  __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);      // 放行常驻核
  auto t0 = std::chrono::steady_clock::now();
  while (cudaStreamQuery(s_persist) != cudaSuccess &&
         std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // ── 共存复现(坑 #38):**消费者核在等待时,别的流还能不能干活?**
  // 现场铁证:kdesc 交付一开,s_tx/s_poll 双双 NotReady 永不恢复
  // (门的三种实现、有无门、legacy/非 legacy 流、通道数、PCIe 降频全试过,
  //  数字逐字节相同)⇒ 元凶锁定在"消费者核在跑"这一件事上。这里用最小
  // 系统复现它:等待中的自旋核 + 另一条流的 D2H memcpy + 第三条流的 launch。
  {
    printf("== 共存:自旋核等待中,其它流能否推进 ==\n");
    *flag = 0; __sync_synchronize();
    unsigned long long a = (unsigned long long)(uintptr_t)flag_d;
    unsigned int w = 1;
    void* p[2] = {&a, &w};
    // 与消费者核同形:4 block × 256 线程(单 block 版已知无害,见上)
    if (launch(fn, 4, 1, 1, 256, 1, 1, 0, s_persist, p, nullptr) != 0) {
      fprintf(stderr, "coexist launch fail\n"); return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cudaStream_t s2;
    CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cudaEvent_t e2;
    CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    cudaMemcpyAsync(hbuf.data(), dbuf, 4096, cudaMemcpyDeviceToHost, s2);
    cudaEventRecord(e2, s2);
    auto t1 = std::chrono::steady_clock::now();
    bool ok2 = false;
    while (std::chrono::steady_clock::now() - t1 < std::chrono::seconds(3)) {
      if (cudaEventQuery(e2) == cudaSuccess) { ok2 = true; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    printf("  D2H@其它流(4 block 自旋核在场):%s\n", ok2 ? "完成 ✓" : "**NotReady ✗**");
    __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
    cudaStreamSynchronize(s_persist);
    cudaStreamDestroy(s2);
    cudaEventDestroy(e2);
    if (!ok2) {
      printf("  ⇒ 复现成功:自旋核在场时其它流停摆(现场同款)\n");
      return 0;
    }
    printf("  ⇒ 未复现:最小系统里其它流正常 ⇒ 现场还有别的变量\n");
  }

  printf("\n=== 判决 ===\n");
  if (hung)
    printf("PERSIST-SYNC DEADLOCK:常驻核在场时 **%s** 挂死\n"
           "⇒ 常驻核方案在 torch 进程内不可行(torch 分配器随时会调它),\n"
           "  必须改有界生命期核(每 collective 起停)或独立 CUDA context。\n", hung);
  else
    printf("NO-DEADLOCK:全部探针在常驻核在场时正常返回 ⇒ 假设被推翻,回查 torch 侧。\n");
  return 0;
}
