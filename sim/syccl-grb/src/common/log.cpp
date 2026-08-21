// Copyright (c) 2025, Alibaba Cloud and its affiliates
// SPDX-License-Identifier: Apache-2.0

#include "common/log.h"
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>

std::string format(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);

  if (std::string(fmt) == "%s") {
    va_end(args);
    return fmt;
  }

  std::vector<char> str(100, '\0');
  int n = vsnprintf(str.data(), str.size(), fmt, args);
  va_end(args);

  if (n < 0) {
    return "Error formatting string!";
  }

  if (n >= str.size()) {
    str.resize(n + 1);
    va_start(args, fmt);
    vsnprintf(str.data(), str.size(), fmt, args);
    va_end(args);
  }

  return std::string(str.data());
}

std::string getCurrentTimestampWithMilliseconds() {
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto value = now_ms.time_since_epoch();
  auto duration = value.count();

  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm tm_time = *std::localtime(&now_c);

  std::ostringstream oss;
  oss << std::put_time(&tm_time, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << (duration % 1000);
  return oss.str();
}

std::string mapToStr(const std::map<int, int> &m) {
  std::string res = "{";
  for (auto &[k, v] : m) {
    res += format("%d->%d,", k, v);
  }
  return res + "}";
}

std::string setToStr(const std::set<int> &s) {
  std::string res = "{";
  for (auto x : s) {
    res += format("%d,", x);
  }
  return res + "}";
}

std::string vecToStr(const std::vector<int> &v) {
  std::string res = "{";
  for (auto x : v) {
    res += format("%d,", x);
  }
  return res + "}";
}
