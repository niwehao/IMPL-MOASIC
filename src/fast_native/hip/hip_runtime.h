/* hip_runtime.h —— 编译 shim(不是 HIP)。
 *
 * FAST 官方 host 侧调度源码(FAST/amd/src/fast_alltoall/*.cc)顶部
 * #include <hip/hip_runtime.h>,但实际代码里所有 hip* 调用都被官方注释掉了
 * (全部换成了 malloc/memcpy)。为了做到「官方源码一行不改」地在无 ROCm 的
 * 机器上编译,这里提供一个空的同名头文件,通过 -I 注入。
 * 见 Imp_Plan/04 §1.3 公平性边界:允许的只有构建 shim 与 adapter。 */
#pragma once
#include <sys/types.h>   /* uint —— 官方代码依赖 glibc 的 uint typedef */
#include <cstdint>       /* uint64_t —— 真 hip_runtime.h 会传递引入,shim 补齐 */
#include <cstring>       /* memset/memcpy —— 同上 */
#include <cstdlib>       /* malloc/free —— 同上 */
