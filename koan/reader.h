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

#ifndef KOAN_READER_H
#define KOAN_READER_H

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "def.h"
#include "indexmap.h"
#include "util.h"

namespace koan {

/// Read lines from a text file and process each using function f.  Each
/// separate sequence (e.g., sentence/paragraph) should be separated by a
/// newline.
///
/// @param[in] fname path to dataset to read
/// @param[in] f function to process each line of input file
/// @tparam: F is a callable on const(std::string_view&).
template <typename F>
inline void readlines(const std::string& fname, F f) {
  FILE* infile = fopen(fname.c_str(), "r");
  KOAN_ASSERT(infile != nullptr,
              "Could not open input file '" + fname +
                  "' -- make sure it exists.");
  std::unique_ptr<char[]> line_c_str(new char[MAX_LINE_LEN]());
  while (fgets(line_c_str.get(), MAX_LINE_LEN, infile) != nullptr) {
    auto line = std::string_view(line_c_str.get());
    KOAN_ASSERT(line.back() == '\n',
                "No end-of-line char! A line in input "
                "data might be too long!");
    line.remove_suffix(1); // remove \n
    f(line);
  }
  fclose(infile);
}

/// Abstract class for reading from a pre-tokenized file.
class Reader {
 protected:
  bool discard_; // discard OOV words instead of replacing with UNK
  std::string fname_;

  // buffers reused to avoid wasteful allocs
  std::vector<std::string_view> words_;

  IndexMap<std::string_view>& word_map_;

  /// Split a sequence into tokens by space.  Handle out-of-vocabulary words
  /// based on the discard flag.
  ///
  /// @param[in] line string_view of a line in the input file.  Corresponds to a
  /// single sequence.
  /// @returns a vector of token indices for this line
  Sentence parseline(const std::string_view& line) {
    Sentence s;

    words_.clear();
    split(words_, line, ' ');

    s.reserve(words_.size());
    for (size_t t = 0; t < words_.size(); t++) {
      const auto index = word_map_.find(words_[t]);

      if (word_map_.is_end(index)) {
        if (not discard_) { s.push_back(word_map_.lookup(UNK)); }
      } else {
        s.push_back(index->second);
      }
    }
    return s;
  }

 public:
  ///
  /// @param[in] word_map vocabulary
  /// @param[in] fname input file path
  /// @param[in] discard flag to toggle between discarding OOV words or
  /// replacing them with UNK
  Reader(IndexMap<std::string_view>& word_map, std::string fname, bool discard)
      : discard_(discard), fname_(fname), word_map_(word_map) {
    words_.reserve(100);
  }
  virtual ~Reader() = default;

  virtual bool get_next(Sentences&) = 0;
};

/// Reader used when one can store the entire training set in memory.
class OnceReader : public Reader {
 private:
  bool read_ = false;
  bool fake_reached_eof_ = false;

 public:
  using Reader::Reader;

  /// Read everything once at the first call, otherwise do nothing as sentences
  /// are already populated.
  ///
  /// @param[in] s list of sentences to be populated
  /// @returns whether we actually read from the file (the first call)
  bool get_next(Sentences& s) override {
    if (not read_) {
      readlines(fname_, [&](const std::string_view& line) {
        s.push_back(parseline(line));
      });
      read_ = true;
    }
    fake_reached_eof_ = not fake_reached_eof_;
    return fake_reached_eof_;
  }
};

/// A reader to be used when you cannot store the entire training set in memory.
class AsyncReader : public Reader {
 private:
  size_t buffer_size_;

  FILE* in_ = nullptr; // filestream to know where we left off
  std::unique_ptr<char[]> line_c_str_ = nullptr;
  Sentences read_buffer_;

  std::unique_ptr<std::thread> reader_;
  bool reached_eof_ = false; // reached EOF in current call to get_next().
  bool reached_eof_prev_ =
      false; // reached EOF in previous call to get_next(),
             //   it needs to return false to reset the loop,
             //   similar to std::getline(ifstream, line).

 public:
  ///
  /// @param[in] word_map vocabulary
  /// @param[in] fname input file path
  /// @param[in] buffer_size number of lines to read into memory at once
  /// @param[in] discard flag to toggle between discarding OOV words or
  /// replacing them with UNK
  AsyncReader(IndexMap<std::string_view>& word_map,
              std::string fname,
              size_t buffer_size,
              bool discard)
      : Reader(word_map, fname, discard), buffer_size_(buffer_size) {
    in_ = fopen(fname_.c_str(), "r");
    line_c_str_ = std::unique_ptr<char[]>(new char[MAX_LINE_LEN]());
    start_reader();
  }

  ~AsyncReader() {
    join_reader();
    fclose(in_);
  }

  /// Initialize reader by populating the line buffer.
  void start_reader() {
    read_buffer_.clear();
    read_buffer_.reserve(buffer_size_);
    reached_eof_ = false;
    reader_ = std::make_unique<std::thread>([this]() {
      while (read_buffer_.size() < buffer_size_) {
        reached_eof_ = fgets(line_c_str_.get(), MAX_LINE_LEN, in_) == nullptr;
        if (reached_eof_) {
          // Reset file ptr to beginning
          fclose(in_);
          in_ = fopen(fname_.c_str(), "r");
          break;
        }

        Sentence s = parseline(line_c_str_.get());
        read_buffer_.push_back(std::move(s));
      }
    });
  }

  void join_reader() { reader_->join(); }

  bool get_next(Sentences& s) override {
    // We want to return false when we cannot read at *current* invocation,
    // which means we reached EOF in previous invocation. reached_eof_prev_
    // keeps track of that.
    if (reached_eof_prev_) {
      reached_eof_prev_ = false;
      return false;
    }

    join_reader();

    reached_eof_prev_ = reached_eof_;
    s = std::move(read_buffer_);
    read_buffer_ = Sentences();

    // While returning the batch of sentences, also immediately start reading
    // the next batch (read_buffer_) in the background
    start_reader();

    return true;
  }
};

} // namespace koan

#endif
