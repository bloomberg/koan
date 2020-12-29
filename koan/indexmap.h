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

#ifndef KOAN_INDEXMAP_H
#define KOAN_INDEXMAP_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "def.h"

namespace koan {

/// Used to store vocabulary map from words to index, and the reverse.
template <typename Key>
class IndexMap {
 private:
  std::unordered_map<Key, size_t> k2i_;
  std::vector<Key> i2k_;

 public:
  IndexMap() {
    k2i_.reserve(INITIAL_INDEX_SIZE);
    i2k_.reserve(INITIAL_INDEX_SIZE);
  }
  IndexMap(const std::unordered_set<Key>& keys) {
    k2i_.reserve(INITIAL_INDEX_SIZE);
    i2k_.reserve(INITIAL_INDEX_SIZE);
    for (const auto& key : keys) { i2k_.push_back(key); }
    for (size_t i = 0; i < i2k_.size(); i++) { k2i_[i2k_[i]] = i; }
  }

  void insert(const Key& key) {
    auto elt = k2i_.emplace(key, i2k_.size());
    if (elt.second) { i2k_.push_back(key); }
  }

  const std::vector<Key>& keys() const { return i2k_; }

  bool has(const Key& key) const { return k2i_.find(key) != k2i_.end(); }

  size_t size() const { return i2k_.size(); }

  void clear() {
    k2i_.clear();
    i2k_.clear();
  }

  typename std::unordered_map<Key, size_t>::const_iterator
  find(const Key& key) const {
    return k2i_.find(key);
  }
  bool
  is_end(typename std::unordered_map<Key, size_t>::const_iterator index) const {
    return index == k2i_.end();
  }
  size_t lookup(const Key& key) const { return k2i_.at(key); }
  size_t operator[](const Key& key) const { return lookup(key); }

  const Key& reverse_lookup(size_t i) const { return i2k_.at(i); }
  const Key& operator()(size_t i) const { return reverse_lookup(i); }
};

} // namespace koan

#endif
