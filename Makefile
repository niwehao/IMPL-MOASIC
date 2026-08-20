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

SRC   := src/util.cc src/topo.cc src/net.cc src/engine.cc src/verbs.cc src/api.cc \
         src/erpc_ctrl.cc
OBJ   := $(SRC:.cc=.o)
BUILD := build

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

$(BUILD)/libmosaic.so: $(OBJ) | $(BUILD)
	$(CXX) -shared -o $@ $(OBJ) $(LDFLAGS) $(ERPC_LDFLAGS)

$(BUILD)/bench_a2a: bench/bench_a2a.cc $(BUILD)/libmosaic.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lmosaic $(LDFLAGS) -Wl,-rpath,$(abspath $(BUILD))

$(BUILD)/test_topo: tests/test_topo.cc src/topo.cc src/util.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -L$(CUDA_ROOT)/lib64 -lcudart -Wl,-rpath,$(CUDA_ROOT)/lib64

# ── 控制面兼容档:eRPC over DPDK(erpc_ctrl.cc)。
# 自动档 —— eRPC 源码 + libnuma 头 + DPDK pkg-config 三者齐备就编入。用自动档
# 而不是手动开关,是为了防两机构建面分叉:一边编入一边没编入时,同一个
# MOSAIC_CTRL 在两机行为不同,现象是会话单向连不上、极难定位。
# 编入 ≠ 启用:运行时仍由能力探测 / MOSAIC_CTRL 选档(见 engine.cc)。
# 强制关掉编入:make ERPC=0。
#
# DPDK 版本钉 21.11:eRPC 的 DpdkTransport 用的是 21.11 及更早的 EAL API 名。
# 链接用动态库 + 绝对路径 rpath —— DPDK 程序要 setcap cap_net_raw,而 setcap
# 后内核进 secure-execution mode 会清空 LD_LIBRARY_PATH;绝对路径 rpath 不受
# 影响,是这里唯一能同时满足「库找得到」和「有 cap」的链接方式。
# eRPC 上游源码是独立 clone(不在本仓库),须先打 third_party/erpc_raven.patch ——
# 不打补丁编出来的 eRPC 在非 root / 无 hugepage / 与内核网络共存的环境下跑不起来。
# 重建步骤写在那个补丁文件的头部。
ERPC_DIR    ?= $(abspath ../../Doc/Code/eRPC)
DPDK_PREFIX ?= $(HOME)/dpdk/install-21.11
# pkgconfig 目录名按发行版而异,现探不写死:Debian 系是 lib/<triplet>/,
# RHEL 系(Raven)是 lib64/。写死一个的后果是另一台机器上自动档静默探不到,
# 于是那台编 stub、这台编完整版 —— 两机构建面分叉,而且不报错(踩过,坑 #52c)。
DPDK_PCDIR  := $(firstword $(wildcard $(DPDK_PREFIX)/lib/x86_64-linux-gnu/pkgconfig \
                                      $(DPDK_PREFIX)/lib64/pkgconfig \
                                      $(DPDK_PREFIX)/lib/pkgconfig))
ERPC ?= $(shell test -f $(ERPC_DIR)/src/rpc.h -a -f /usr/include/numa.h \
                     -a -f $(DPDK_PCDIR)/libdpdk.pc && echo 1 || echo 0)
ifeq ($(ERPC),1)
ERPC_CONFIG := $(BUILD)/erpc-dpdk/config
DPDK_CFLAGS := $(shell PKG_CONFIG_PATH=$(DPDK_PCDIR) pkg-config --cflags libdpdk)
# **静态**链接 DPDK。三个理由,缺一不可:
#   ① setcap 后进 secure-execution mode,LD_LIBRARY_PATH 被内核清空;
#   ② DPDK 自己的 .so 没有 RUNPATH,库与库之间互相找不到(而 RUNPATH 不传递,
#      libmosaic.so 的 rpath 只管得到它的直接依赖);
#   ③ whole-archive 是 PMD 必需的 —— 驱动靠 RTE_INIT 构造函数自注册,
#      按需链接会把整个 mlx5 PMD 丢掉,现象是 EAL 起得来但探不到任何端口。
DPDK_LIBS   := $(shell PKG_CONFIG_PATH=$(DPDK_PCDIR) pkg-config --static --libs libdpdk)
# eRPC 的编译期常量(对应上游 CMakeLists 的 TRANSPORT=dpdk 分支):
#   CTransport=DpdkTransport、kHeadroom=40(DPDK 以太头留白)、非 RoCE、非 Azure。
ERPC_DEFS := -DERPC_DPDK=true -DERPC_LOG_LEVEL=2 -DERPC_TESTING=false
# **只**给 eRPC 相关编译单元的 flags。绝不并进全局 CXXFLAGS —— DPDK 的
# cflags 带 -march=corei7,并进去会改变主体代码的指令选择,那就不是
# 「只动兼容档这条线」了。
ERPC_INC := $(ERPC_DEFS) -I$(ERPC_DIR)/src -I$(ERPC_DIR)/third_party/asio/include \
            -I$(ERPC_CONFIG) $(DPDK_CFLAGS)
CXXFLAGS_ERPC := $(CXXFLAGS) -DMOSAIC_HAVE_ERPC $(ERPC_INC)
ERPC_LDFLAGS := $(DPDK_LIBS) -lnuma

ERPC_SRCS := nexus_impl/nexus nexus_impl/nexus_bg_thread nexus_impl/nexus_sm_thread \
             rpc_impl/rpc rpc_impl/rpc_queues rpc_impl/rpc_rfr rpc_impl/rpc_cr \
             rpc_impl/rpc_kick rpc_impl/rpc_req rpc_impl/rpc_resp rpc_impl/rpc_ev_loop \
             rpc_impl/rpc_fault_inject rpc_impl/rpc_pkt_loss rpc_impl/rpc_rx \
             rpc_impl/rpc_connect_handlers rpc_impl/rpc_disconnect_handlers \
             rpc_impl/rpc_reset_handlers rpc_impl/rpc_sm_api rpc_impl/rpc_sm_helpers \
             transport_impl/transport transport_impl/dpdk/dpdk_transport \
             transport_impl/dpdk/dpdk_transport_datapath \
             transport_impl/dpdk/dpdk_init transport_impl/dpdk/dpdk_externs \
             util/huge_alloc util/numautils util/tls_registry
ERPC_OBJ := $(addprefix $(BUILD)/erpc-dpdk/obj/,$(addsuffix .o,$(ERPC_SRCS)))

$(ERPC_CONFIG)/config.h: Makefile | $(BUILD)
	@mkdir -p $(ERPC_CONFIG)
	@printf '#pragma once\n#include <stdlib.h>\nnamespace erpc {\nclass IBTransport;\nclass RawTransport;\nclass DpdkTransport;\nclass FakeTransport;\n#define CTransport DpdkTransport\nstatic constexpr size_t kHeadroom = 40;\nstatic constexpr size_t kIsRoCE = false;\nstatic constexpr size_t kIsAzure = false;\n}\n' > $@

$(BUILD)/erpc-dpdk/obj/%.o: $(ERPC_DIR)/src/%.cc $(ERPC_CONFIG)/config.h | $(BUILD)
	@mkdir -p $(dir $@)
	$(CXX) -O2 -g -std=c++17 -w -fPIC -MMD -MP $(ERPC_INC) -c $< -o $@

# erpc_ctrl.cc 是唯一需要 eRPC 头的自家源文件,单独一条规则带 eRPC flags
src/erpc_ctrl.o: src/erpc_ctrl.cc $(ERPC_CONFIG)/config.h
	$(CXX) $(CXXFLAGS_ERPC) -c $< -o $@

# 注:libmosaic.so 的前置依赖在规则行是立即展开的,这里追加依赖 + OBJ
# (链接 recipe 里的 $(OBJ) 是延迟展开,能看到追加后的值)
OBJ += $(ERPC_OBJ)
$(BUILD)/libmosaic.so: $(ERPC_OBJ)
-include $(ERPC_OBJ:.o=.d)
endif

clean:
	rm -f $(OBJ) $(OBJ:.o=.d); rm -rf $(BUILD)

-include $(OBJ:.o=.d)

.PHONY: all clean


