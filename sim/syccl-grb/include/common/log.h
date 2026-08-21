// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdarg>
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <mutex>

#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_TRACE 2
#define LOG_LEVEL_DEBUG 3

#define LOG_LEVEL_ALGO LOG_LEVEL_INFO
#define LOG_LEVEL_PERFMODEL LOG_LEVEL_INFO
#define LOG_LEVEL_SOLVER LOG_LEVEL_INFO

#define LOG_ENABLED(subsys_level, level) (subsys_level >= level)

std::string format(const char* fmt, ...);

std::string getCurrentTimestampWithMilliseconds();

inline const char* get_filename_from_path(const char *path) {
  const char *file = strrchr(path, '/');
  if (file)
    return file + 1;
  else
    return path;
}

#define LOOOG_INTERNAL(file, line, subsys, level, fmt, ...) do { \
    if (LOG_ENABLED(LOG_LEVEL_##subsys, LOG_LEVEL_##level)) { \
        std::cout << "[" <<  getCurrentTimestampWithMilliseconds() << "]"; \
        std::cout << "[" << file << ":" << line << "]"; \
        std::cout << "[" #subsys "][" #level "] "; \
        std::cout << format(fmt, ##__VA_ARGS__) << std::endl; \
    } \
} while (0)

#define LOOOG_SIMPLE_INTERNAL(file, line, subsys, level, fmt, ...) do { \
    if (LOG_ENABLED(LOG_LEVEL_##subsys, LOG_LEVEL_##level)) { \
        std::cout << format(fmt, ##__VA_ARGS__) << std::endl; \
    } \
} while (0)

#define ENABLE_GLOBAL_LOG
// #undef ENABLE_GLOBAL_LOG

#ifdef ENABLE_GLOBAL_LOG
#define LOOOG(subsys, level, fmt, ...) \
    LOOOG_INTERNAL(get_filename_from_path(__FILE__), __LINE__, subsys, level, fmt, ##__VA_ARGS__)

#define LOG(subsys, level, fmt, ...) \
    LOOOG_SIMPLE_INTERNAL(get_filename_from_path(__FILE__), __LINE__, subsys, level, fmt, ##__VA_ARGS__)

#else 

#define LOOOG(subsys, level, fmt, ...)
#define LOG(subsys, level, fmt, ...)

#endif

std::string mapToStr(const std::map<int, int> &m);

std::string setToStr(const std::set<int> &s);

std::string vecToStr(const std::vector<int> &v);

inline std::mutex &getGlobalLogMutex() {
  static std::mutex global_log_mutex;
  return global_log_mutex;
}