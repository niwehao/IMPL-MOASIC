cmake_minimum_required(VERSION 3.16)
project(mosaic_rt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")

# CUDA 只用 runtime API(数据面全是 memcpy,没有自定义 kernel),
# 所以用普通 g++ 编译 + 链 libcudart 即可,不需要 nvcc。
# 机器上 /usr/local/cuda-13.3 只有 runtime lib,用户态装的 12.8 才有完整 include。
set(CUDA_ROOT "$ENV{HOME}/cuda-12.8" CACHE PATH "CUDA toolkit root")

add_library(mosaic SHARED
  src/util.cc
  src/topo.cc
  src/net.cc
  src/engine.cc
  src/fast_sched.cc
  src/api.cc
)
target_include_directories(mosaic PUBLIC include PRIVATE ${CUDA_ROOT}/include)
target_link_directories(mosaic PRIVATE ${CUDA_ROOT}/lib64)
target_link_libraries(mosaic PRIVATE cudart rt pthread)
set_target_properties(mosaic PROPERTIES INSTALL_RPATH "${CUDA_ROOT}/lib64" BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(bench_a2a bench/bench_a2a.cc)
target_link_libraries(bench_a2a PRIVATE mosaic)
target_include_directories(bench_a2a PRIVATE ${CUDA_ROOT}/include include)
target_link_directories(bench_a2a PRIVATE ${CUDA_ROOT}/lib64)
target_link_libraries(bench_a2a PRIVATE cudart)
set_target_properties(bench_a2a PROPERTIES INSTALL_RPATH "${CUDA_ROOT}/lib64" BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(test_topo tests/test_topo.cc src/topo.cc src/util.cc)
target_include_directories(test_topo PRIVATE include src ${CUDA_ROOT}/include)
target_link_libraries(test_topo PRIVATE pthread)
