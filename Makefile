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
         src/fast_native.cc src/api.cc src/erpc_ctrl.cc
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
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread

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

# ── eRPC 控制面(S0d,可选;源码 = 根目录 Code/eRPC,补丁见
# common/raven/patches/erpc_raven.patch)。与 NVSHMEM 同款自动档:目录在
# 且有 libnuma 头就编入(防两机构建面分叉,log.md 坑 #21 教训);
# 运行时仍要 MOSAIC_CTRL=erpc 才启用。transport 固定 infiniband,
# RoCE 与否按本机 HCA link_layer 自动判(本机 100GbE RoCE / Raven IB HDR)。
ERPC_DIR ?= $(shell test -f $(abspath ../../Code/eRPC)/src/rpc.h && echo $(abspath ../../Code/eRPC) || echo $(HOME)/Code/eRPC)
ERPC ?= $(shell test -f $(ERPC_DIR)/src/rpc.h -a -f /usr/include/numa.h && echo 1 || echo 0)
ifeq ($(ERPC),1)
ERPC_CONFIG := $(BUILD)/erpc_config
ERPC_ROCE ?= $(shell grep -q InfiniBand /sys/class/infiniband/*/ports/*/link_layer 2>/dev/null && echo 0 || echo 1)
ifeq ($(ERPC_ROCE),1)
ERPC_HEADROOM := 40
ERPC_ISROCE := true
else
ERPC_HEADROOM := 0
ERPC_ISROCE := false
endif
ERPC_DEFS := -DERPC_INFINIBAND=true -DERPC_LOG_LEVEL=2 -DERPC_TESTING=false
CXXFLAGS += -DMOSAIC_HAVE_ERPC $(ERPC_DEFS) -I$(ERPC_DIR)/src -I$(ERPC_DIR)/third_party/asio/include -I$(ERPC_CONFIG)
LDFLAGS  += -lnuma
ERPC_SRCS := nexus_impl/nexus nexus_impl/nexus_bg_thread nexus_impl/nexus_sm_thread \
             rpc_impl/rpc rpc_impl/rpc_queues rpc_impl/rpc_rfr rpc_impl/rpc_cr \
             rpc_impl/rpc_kick rpc_impl/rpc_req rpc_impl/rpc_resp rpc_impl/rpc_ev_loop \
             rpc_impl/rpc_fault_inject rpc_impl/rpc_pkt_loss rpc_impl/rpc_rx \
             rpc_impl/rpc_connect_handlers rpc_impl/rpc_disconnect_handlers \
             rpc_impl/rpc_reset_handlers rpc_impl/rpc_sm_api rpc_impl/rpc_sm_helpers \
             transport_impl/transport transport_impl/infiniband/ib_transport \
             transport_impl/infiniband/ib_transport_datapath \
             util/huge_alloc util/numautils util/tls_registry
ERPC_OBJ := $(addprefix $(BUILD)/erpc/,$(addsuffix .o,$(ERPC_SRCS)))

$(ERPC_CONFIG)/config.h: | $(BUILD)
	@mkdir -p $(ERPC_CONFIG)
	@printf '#pragma once\n#include <stdlib.h>\nnamespace erpc {\nclass IBTransport;\nclass RawTransport;\nclass DpdkTransport;\nclass FakeTransport;\n#define CTransport IBTransport\nstatic constexpr size_t kHeadroom = $(ERPC_HEADROOM);\nstatic constexpr size_t kIsRoCE = $(ERPC_ISROCE);\nstatic constexpr size_t kIsAzure = false;\n}\n' > $@

$(BUILD)/erpc/%.o: $(ERPC_DIR)/src/%.cc $(ERPC_CONFIG)/config.h | $(BUILD)
	@mkdir -p $(dir $@)
	$(CXX) -O2 -g -std=c++17 -w -fPIC -MMD -MP $(ERPC_DEFS) -I$(ERPC_DIR)/src -I$(ERPC_DIR)/third_party/asio/include -I$(ERPC_CONFIG) -c $< -o $@

src/erpc_ctrl.o: $(ERPC_CONFIG)/config.h
# 注:libmosaic.so 的前置依赖在规则行是立即展开的,这里追加依赖 + OBJ
# (链接 recipe 里的 $(OBJ) 是延迟展开,能看到追加后的值)
OBJ += $(ERPC_OBJ)
$(BUILD)/libmosaic.so: $(ERPC_OBJ)
-include $(ERPC_OBJ:.o=.d)
endif
