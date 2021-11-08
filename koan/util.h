/*
** Copyright 2020 Bloomberg Finance L.P.
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

#ifndef KOAN_UTIL_H
#define KOAN_UTIL_H

#include <algorithm>
#include <atomic>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace koan {

std::string date_time(const std::string& format) {
  std::string ret(50, char());
  std::time_t tt = std::time(nullptr);
  ret.resize(std::strftime(
      ret.data(), ret.size(), format.c_str(), std::localtime(&tt)));
  return ret;
}

template <typename OUT, typename IN>
void split(std::vector<OUT>& ret, const IN& s, char delim = ' ') {
  auto beg = s.begin();
  while (beg < s.end()) {
    auto end = std::find(beg, s.end(), delim);
    if (beg != end) { ret.emplace_back(&*beg, end - beg); }
    beg = ++end;
  }
}

template <typename OUT, typename IN>
auto split(const IN& s, char delim = ' ') {
  std::vector<OUT> ret;
  split(ret, s, delim);
  return ret;
}

/// Parallel for implementation without any explicit allocation of elements per
/// thread.
///
/// @param[in] begin start index
/// @param[in] end end index
/// @param[in] f function to process each element
/// @param[in] num_threads number of threads to run
/// @tparam F callable that takes size_t elt_idx, size_t thread_idx as arguments
template <typename F>
void parallel_for(size_t begin, size_t end, F f, size_t num_threads = 8) {
  std::vector<std::thread> threads(num_threads);
  std::atomic<size_t> i = begin;
  for (size_t ti = 0; ti < num_threads; ti++) {
    auto& t = threads[ti];
    t = std::thread([ti, &i, &f, &end]() {
      while (true) {
        size_t i_ = i++;
        if (i_ >= end) { break; }
        f(i_, ti);
      }
    });
  }

  for (auto& t : threads) t.join();
}

/// Parallel for implementation where each thread is allotted its own batch of
/// elements to process up front.
///
/// @param[in] begin start index
/// @param[in] end end index
/// @param[in] f function to process each element
/// @param[in] num_threads number of threads to run
/// @param[in] consecutive_alloc if true, allocate a contiguous block of
/// elements to each thread
/// @tparam F callable that takes size_t elt_idx, size_t thread_idx as arguments
template <typename F>
void parallel_for_partitioned(size_t begin,
                              size_t end,
                              F f,
                              size_t num_threads = 8,
                              bool consecutive_alloc = true) {
  size_t total_size = end - begin;
  size_t batch_size = total_size / num_threads;
  std::vector<std::thread> threads(num_threads);
  for (size_t ti = 0; ti < num_threads; ti++) {
    auto& t = threads[ti];
    if (consecutive_alloc) {
      t = std::thread([ti, &f, begin, end, batch_size, num_threads]() {
        size_t batch_start = begin + ti * batch_size;
        size_t batch_end =
            ti < (num_threads - 1) ? begin + (ti + 1) * batch_size : end;
        for (size_t i = batch_start; i < batch_end; ++i) { f(i, ti); }
      });
    } else {
      t = std::thread([ti, &f, begin, end, num_threads]() {
        for (size_t i = begin + ti; i < end; i += num_threads) { f(i, ti); }
      });
    }
  }

  for (auto& t : threads) { t.join(); }
}

class RuntimeError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};

}; // namespace koan

#define KOAN_OVERLOAD(_1, _2, MACRO, ...) MACRO

#define KOAN_ASSERT(...)                                                       \
  KOAN_OVERLOAD(__VA_ARGS__, KOAN_ASSERT2, KOAN_ASSERT1)(__VA_ARGS__)

#define KOAN_ASSERT2(statement, message)                                       \
  if (!(statement)) { throw koan::RuntimeError(message); }

#define KOAN_ASSERT1(statement)                                                \
  KOAN_ASSERT2(statement, "Assertion " #statement " failed!")

#endif
