# mosaic-rt 构建。本机没有 cmake(只有 g++/make),所以 Makefile 是主构建入口;
# CMakeLists.txt 留给 Raven(那边 module load 有 cmake)。
#
# 关键点:数据面没有自定义 CUDA kernel(全是 memcpy + 偏移算术),所以整个库用
# 普通 g++ 编译、只链 libcudart 就行 —— 不需要 nvcc,也就绕开了本机
# glibc 2.41 与 CUDA 12.8 头文件的 cospi/sinpi 冲突(见 log.md 坑 #2)。

CUDA_ROOT ?= $(HOME)/cuda-12.8
CXX       ?= g++
# -MMD -MP:头文件依赖跟踪。**不能省** —— 少了它,改 internal.h 后只有被直接
# 编辑过的 .cc 会重编,其余 .o 仍按**旧的 struct 布局**访问 Comm,写进错位的偏移。
# 症状是随机的堆损坏 / glibc mutex 断言,完全无法从症状反推(实测踩过,见 log.md 坑 #4)。
CXXFLAGS  := -O2 -g -std=c++17 -msse4.2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable \
             -MMD -MP -fPIC -Iinclude -Isrc -I$(CUDA_ROOT)/include
LDFLAGS   := -L$(CUDA_ROOT)/lib64 -lcudart -libverbs -lrt -lpthread -Wl,-rpath,$(CUDA_ROOT)/lib64

SRC   := src/util.cc src/topo.cc src/net.cc src/engine.cc src/fast_sched.cc src/verbs.cc src/nvshmem_pipe.cc \
         src/fast_native.cc src/api.cc
OBJ   := $(SRC:.cc=.o)
BUILD := build
.DEFAULT_GOAL := all   # 下面的 FAST 规则写在 all 之前,必须显式钉住默认目标

# ── FAST 官方 host 调度器(零拷贝、零修改,直接从官方仓库路径编译)──
# 公平性铁律(Imp_Plan/04 §1.3):官方源码一行不改;hip 头用空 shim 注入
# (src/fast_native/hip/hip_runtime.h),-DNOLOG 只是关闭官方自己的 printf 宏
# (官方 alltoall_define.h 提供的开关,不是修改)。.o 落在 build/ 下,
# 不污染官方目录。
FAST_DIR := $(abspath ../fast-rt/FAST/amd)
FAST_INC := -I$(FAST_DIR)/src/include -Isrc/fast_native
FAST_SRCS := alltoall_matrix alltoall_algorithm alltoall_local_scheduler alltoall_global_scheduler
FAST_OBJ  := $(addprefix $(BUILD)/fast_official/,$(addsuffix .o,$(FAST_SRCS)))

# -include hip/hip_runtime.h 模拟 hipcc 的预包含行为(官方源文件把自家头放在
# hip 头之前,靠 hipcc 预包含拿到 uint64_t;g++ 下不加这个会编译不过)
$(BUILD)/fast_official/%.o: $(FAST_DIR)/src/fast_alltoall/%.cc | $(BUILD)
	@mkdir -p $(BUILD)/fast_official
	$(CXX) -O2 -g -std=c++17 -w -fPIC -DNOLOG $(FAST_INC) -include hip/hip_runtime.h -c $< -o $@

src/fast_native.o: src/fast_native.cc
	$(CXX) $(CXXFLAGS) $(FAST_INC) -c $< -o $@

all: $(BUILD)/libmosaic.so $(BUILD)/bench_a2a $(BUILD)/test_topo $(BUILD)/replay_core_test $(BUILD)/test_kcons

$(BUILD)/replay_core_test: tests/replay_core_test.cc $(BUILD)/libmosaic.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lmosaic $(LDFLAGS) -Wl,-rpath,$(abspath $(BUILD))

# S3 二期消费者内核独立单测(单 GPU,无网络;JIT 同一份嵌入 PTX)
$(BUILD)/test_kcons: tests/test_kcons.cc src/consumer_kernel_ptx.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# 平台探针:常驻核在场时哪些 host CUDA API 死锁(坑 #35 归因;不进 regression)
$(BUILD)/test_persist_sync: tests/test_persist_sync.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD):
	@mkdir -p $(BUILD)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/libmosaic.so: $(OBJ) $(FAST_OBJ) | $(BUILD)
	$(CXX) -shared -o $@ $(OBJ) $(FAST_OBJ) $(LDFLAGS)

$(BUILD)/bench_a2a: bench/bench_a2a.cc $(BUILD)/libmosaic.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lmosaic $(LDFLAGS) -Wl,-rpath,$(abspath $(BUILD))

$(BUILD)/test_topo: tests/test_topo.cc src/topo.cc src/util.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -L$(CUDA_ROOT)/lib64 -lcudart -Wl,-rpath,$(CUDA_ROOT)/lib64

clean:
	rm -f $(OBJ) $(OBJ:.o=.d); rm -rf $(BUILD)

-include $(OBJ:.o=.d)

.PHONY: all clean

# ── NVSHMEM 后端(可选;make NVSHMEM=1)──
# 头/库来自 pip wheel nvidia-nvshmem-cu12(无 CMake config,路径直给)。
# host-only API(putmem_signal_nbi_on_stream 等),不编设备内核 → g++ 即可。
NVSHMEM_ROOT ?= $(HOME)/CodeRead/nano-vllm/.venv/lib/python3.12/site-packages/nvidia/nvshmem
# wheel 在就默认编入(运行时仍要 MOSAIC_NIC=nvshmem 才启用)。之前默认关,
# sync.sh 的远端普通 make 会把 .o 编成无 NVSHMEM 版,后续 make NVSHMEM=1
# 又"已最新"不重编 —— 两机构建面悄悄分叉(log.md 坑 #21)。
NVSHMEM ?= $(shell test -d $(NVSHMEM_ROOT)/include && echo 1 || echo 0)
ifeq ($(NVSHMEM),1)
CXXFLAGS += -DMOSAIC_HAVE_NVSHMEM -I$(NVSHMEM_ROOT)/include
LDFLAGS  += $(NVSHMEM_ROOT)/lib/libnvshmem_host.so.3 -lcudart -ldl -Wl,-rpath,$(NVSHMEM_ROOT)/lib
endif

