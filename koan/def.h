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

#ifndef KOAN_DEF_H
#define KOAN_DEF_H

#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace koan {

#ifdef KOAN_GRAD_CHECKING
using Real = double;
#else
using Real = float;
#endif
constexpr Real operator"" _R(long double d) {
  return d;
}
constexpr Real operator"" _R(unsigned long long d) {
  return d;
}

using Vector = Eigen::Matrix<Real, -1, 1>;
using Table = std::vector<Vector>;

using Word = unsigned;
using Sentence = std::vector<Word>;
using Sentences = std::vector<Sentence>;

const static std::string UNKSTR = "___UNK___";
const static std::string_view UNK(UNKSTR);

const static size_t INITIAL_INDEX_SIZE = 30000000;
const static size_t INITIAL_SENTENCE_LEN = 1000;
const static int MAX_LINE_LEN = 1000000;

// based on the first nonzero entry in the sigmoid approx. table
const static Real MIN_SIGMOID_IN_LOSS = 0.000340641;

} // namespace koan

#endif
