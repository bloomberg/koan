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

#ifndef KOAN_SAMPLE_H
#define KOAN_SAMPLE_H

#include <algorithm>
#include <random>
#include <vector>

#include "def.h"
#include "util.h"

namespace koan {

/// Algorithm to sample from a fixed categorical distribution in constant time.
/// Implements Vose's Alias Method as described in:
/// https://www.keithschwarz.com/darts-dice-coins/
class AliasSampler {
 public:
  using Index = size_t;

 private:
  std::vector<Index> alias_; // alias class for each bucket
  std::vector<Real> prob_;   // threshold for selecting the alias class
  std::uniform_int_distribution<Index> macro_dist_;
  std::uniform_real_distribution<Real> micro_dist_;
  std::minstd_rand rng_;
  size_t n_;

  /// Initialize alias table.  Steps correspond to those listed in
  /// "Algorithm: Vose's Alias Method" of
  /// https://www.keithschwarz.com/darts-dice-coins/
  ///
  /// @param[in] probs multinomial distribution to represent
  void init_alias_table(const std::vector<Real>& probs) {
    // Ensure this is a valid probability distribution
    KOAN_ASSERT(std::all_of(
        probs.begin(), probs.end(), [](Real p) { return p >= 0.0; }));
    Real probSum = std::accumulate(probs.begin(), probs.end(), 0.0);
    KOAN_ASSERT((0.9999 <= probSum) and (probSum <= 1.0001));

    // Step 2
    std::vector<Index> small;
    std::vector<Index> large;

    // Step 3
    std::vector<Real> scaledProbs = probs;
    for (size_t i = 0; i < scaledProbs.size(); ++i) { scaledProbs[i] *= n_; }

    // Step 4
    for (size_t i = 0; i < scaledProbs.size(); ++i) {
      Real p_i = scaledProbs[i];

      if (p_i < 1.0) {
        small.push_back(i);
      } else {
        large.push_back(i);
      }
    }

    // Step 5
    Index l;
    Index g;

    while (not(small.empty() or large.empty())) {
      l = small.back();
      g = large.back();
      small.pop_back();
      large.pop_back();

      prob_[l] = scaledProbs[l];
      alias_[l] = g;
      scaledProbs[g] = (scaledProbs[g] + scaledProbs[l]) - 1;
      if (scaledProbs[g] < 1.0) {
        small.push_back(g);
      } else {
        large.push_back(g);
      }
    }

    // Step 6
    while (not large.empty()) {
      g = large.front();
      large.erase(large.begin());
      prob_[g] = 1.0;
    }

    // Step 7
    while (not small.empty()) {
      l = small.front();
      small.erase(small.begin());
      prob_[l] = 1.0;
    }
  }

 public:
  AliasSampler(const std::vector<Real>& probs)
      : alias_(probs.size(), 0),
        prob_(probs.size(), 0.0),
        macro_dist_(1, probs.size()),
        micro_dist_(0.0, 1.0),
        rng_(),
        n_(probs.size()) {
    init_alias_table(probs);
  }

  void set_seed(unsigned seed) { rng_.seed(seed); }

  Index sample() {
    Index bucket = macro_dist_(rng_) - 1;
    Real r = micro_dist_(rng_);
    if (r <= prob_[bucket]) {
      return bucket;
    } else {
      return alias_[bucket];
    }
  }

  size_t num_classes() { return n_; }
};

} // namespace koan

#endif
