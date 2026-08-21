cmake_minimum_required(VERSION 3.18)
project(synthesize CXX)
option(CXX "" ON)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_compile_options(-O3)

# Set the GUROBI_DIR variable if not provided by the user
if(DEFINED ENV{GUROBI_HOME})
    set(GUROBI_DIR $ENV{GUROBI_HOME})
    message("GUROBI_DIR already set: ${GUROBI_DIR}")
else()
    message("GUROBI_DIR not set. Please set the GUROBI_HOME environment variable to your Gurobi installation path.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(GUROBI REQUIRED)

include_directories(include ${GUROBI_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCE_FILES
    src/common/*.cpp
    src/Sketch/*.cpp
    src/PerfModel/*.cpp
    src/Solver/*.cpp
    src/Solver/Gurobi/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES})
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${SRC_LIST})

include_directories(
    include
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  include(FeatureSummary)
  feature_summary(WHAT ALL)
endif()
