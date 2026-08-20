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

SRC   := src/util.cc src/topo.cc src/net.cc src/engine.cc src/verbs.cc src/api.cc
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
	$(CXX) -shared -o $@ $(OBJ) $(LDFLAGS)

$(BUILD)/bench_a2a: bench/bench_a2a.cc $(BUILD)/libmosaic.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(BUILD) -lmosaic $(LDFLAGS) -Wl,-rpath,$(abspath $(BUILD))

$(BUILD)/test_topo: tests/test_topo.cc src/topo.cc src/util.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -L$(CUDA_ROOT)/lib64 -lcudart -Wl,-rpath,$(CUDA_ROOT)/lib64

clean:
	rm -f $(OBJ) $(OBJ:.o=.d); rm -rf $(BUILD)

-include $(OBJ:.o=.d)

.PHONY: all clean


