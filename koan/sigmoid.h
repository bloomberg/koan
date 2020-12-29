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

#ifndef KOAN_SIGMOID_H
#define KOAN_SIGMOID_H

#include <algorithm>
#include <array>
#include "def.h"

namespace koan {

/// Sigmoid.  Defaults to table lookup implementation unless checking gradient
/// numerically.
///
/// @param[in] x logit
/// @returns $\sigma(x)$
Real sigmoid(Real x) {
  // Based on sigmoid(x) == tanh(x/2)/2 + 1/2
  // std::tanh can handle extremes correctly out-of-the-box, i.e.
  // tanh(-Inf) = -1 and tanh(Inf) = 1 instead of Inf or NaN.
#ifdef KOAN_GRAD_CHECKING
  return std::fma(std::tanh(x * .5_R), .5_R, .5_R);
#else
  static constexpr Real factor = 64_R, window = 8_R;
  static const auto table = [&]() {
    std::array<Real, size_t(factor * window * 2_R + 1_R)> ret;
    std::generate(ret.begin(), ret.end(), [i = -factor * window]() mutable {
      return std::fma(std::tanh(i++ / factor * .5_R), .5_R, .5_R);
    });
    ret.front() = 0_R;
    ret.back() = 1_R;
    return ret;
  }();
  static constexpr Real lo = -window, hi = window;
  static constexpr Real m = factor, a = factor * window;
  return table[size_t(std::fma(std::clamp(x, lo, hi), m, a))];
#endif
};

} // namespace koan

#endif
