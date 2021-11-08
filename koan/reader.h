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

#ifdef KOAN_ENABLE_ZIP
#include "zlib.h"
#endif

namespace koan {

/// Abstraction over type of file to train on.
class TrainFileHandler {
 protected:
  const std::string& fname_;

 public:
  TrainFileHandler(const std::string& fname) : fname_(fname) {}

  virtual char* gets(char* buf, int len) = 0;
  virtual void close() = 0;

  virtual ~TrainFileHandler() = default;
};

/// Reads plain text files
class TextFileHandler : public TrainFileHandler {
 private:
  FILE* f;

 public:
  TextFileHandler(const std::string& fname) : TrainFileHandler(fname) {
    f = fopen(fname.c_str(), "r");
    KOAN_ASSERT(f != nullptr,
                "Could not open input file '" + fname +
                    "' -- make sure it exists.");
  }

  char* gets(char* buf, int len) override { return fgets(buf, len, f); }

  void close() override { fclose(f); }
};

#ifdef KOAN_ENABLE_ZIP
/// Reads gzipped files
class GzipFileHandler : public TrainFileHandler {
 private:
  gzFile f;

 public:
  GzipFileHandler(const std::string& fname) : TrainFileHandler(fname) {
    f = gzopen(fname.c_str(), "r");

    KOAN_ASSERT(f != nullptr,
                "Could not open input file '" + fname +
                    "' -- make sure it exists.");
  }

  char* gets(char* buf, int len) override { return gzgets(f, buf, len); }

  void close() override { gzclose(f); }
};
#endif

std::unique_ptr<TrainFileHandler> getfilehandler(const std::string& fname,
                                                 const std::string& read_mode) {

#ifdef KOAN_ENABLE_ZIP
  bool is_ext_gzip =
      fname.size() >= 3 and fname.compare(fname.size() - 3, 3, ".gz") == 0;

  if (read_mode == "gzip" or (is_ext_gzip && read_mode == "auto")) {
    return std::make_unique<GzipFileHandler>(fname);
  }
#endif

  return std::make_unique<TextFileHandler>(fname);
}

/// Read lines from a training file and process each using function f.  Each
/// separate sequence (e.g., sentence/paragraph) should be separated by a
/// newline.
///
/// @param[in] fname path to dataset to read
/// @param[in] f function to process each line of input file
/// @param[in] read_mode how to read from each file.  Respected if compiled with
/// KOAN_ENABLE_ZIP, otherwise assumes all are plain text files.
/// @tparam: F is a callable on const(std::string_view&).
template <typename F>
void readlines(const std::vector<std::string>& fnames,
               F f,
               std::string read_mode,
               bool assert_no_long_lines = false) {
  for (const std::string& fname : fnames) {
    auto fhandler = getfilehandler(fname, read_mode);

    std::unique_ptr<char[]> line_c_str(new char[MAX_LINE_LEN]());
    while (fhandler->gets(line_c_str.get(), MAX_LINE_LEN) != nullptr) {
      auto line = std::string_view(line_c_str.get());

      if (assert_no_long_lines) {
        KOAN_ASSERT(line.back() == '\n',
                    "No end-of-line char! A line in input "
                    "data might be too long in file '" +
                        fname + "'");
      }

      line.remove_suffix(1); // remove \n
      f(line);
    }

    fhandler->close();
  }
}

template <typename F>
void readlines(const std::string& fname,
               F f,
               std::string read_mode,
               bool assert_no_long_lines) {
  const std::vector<std::string> fname_vec{fname};
  readlines(fname_vec, f, read_mode, assert_no_long_lines);
}

/// Abstract class for reading from a pre-tokenized file.
class Reader {
 protected:
  bool discard_;              // discard OOV words instead of replacing with UNK
  bool assert_no_long_lines_; // whether to throw on lines > MAX_LINE_LEN chars

  std::vector<std::string> fnames_;
  std::string read_mode_;

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

      if (index == word_map_.end()) {
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
  /// @param[in] read_mode define behavior for reading from files.  "text":
  /// treat all files as plain text; "gzip": treat all files as gzipped; "auto":
  /// treat *.gz as gzipped, otherwise plain text
  Reader(IndexMap<std::string_view>& word_map,
         std::vector<std::string>& fnames,
         bool discard,
         std::string read_mode,
         bool assert_no_long_lines = false)
      : discard_(discard),
        assert_no_long_lines_(assert_no_long_lines),
        fnames_(fnames),
        read_mode_(read_mode),
        word_map_(word_map) {
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
      readlines(
          fnames_,
          [&](const std::string_view& line) { s.push_back(parseline(line)); },
          read_mode_,
          assert_no_long_lines_);

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

  std::unique_ptr<TrainFileHandler> in_; // handler of current file, track where
                                         // we left off
  size_t path_idx_ = 0; // index into which file we are reading from
  std::unique_ptr<char[]> line_c_str_ = nullptr;
  Sentences read_buffer_;

  std::unique_ptr<std::thread> reader_;
  bool reached_eof_ = false;  // reached EOF in current call to get_next().
  bool reached_eofs_ = false; // reached EOF for the last file in current call
                              // to get_next().
  bool reached_eofs_prev_ = false; // reached EOF in previous call to get_next()
                                   // it needs to return false to reset the
                                   // loop, similar to
                                   // std::getline(ifstream, line).

 public:
  ///
  /// @param[in] word_map vocabulary
  /// @param[in] fname input file path
  /// @param[in] buffer_size number of lines to read into memory at once
  /// @param[in] discard flag to toggle between discarding OOV words or
  /// replacing them with UNK
  AsyncReader(IndexMap<std::string_view>& word_map,
              std::vector<std::string>& fnames,
              size_t buffer_size,
              bool discard,
              const std::string& read_mode,
              bool assert_no_long_lines)
      : Reader(word_map, fnames, discard, read_mode, assert_no_long_lines),
        buffer_size_(buffer_size),
        path_idx_(0) {

    in_ = getfilehandler(fnames_[path_idx_], read_mode_);
    line_c_str_ = std::unique_ptr<char[]>(new char[MAX_LINE_LEN]());
    start_reader();
  }

  ~AsyncReader() {
    join_reader();
    in_->close();
  }

  /// Initialize reader by populating the line buffer.
  void start_reader() {
    read_buffer_.clear();
    read_buffer_.reserve(buffer_size_);
    reached_eofs_ = false;

    reader_ = std::make_unique<std::thread>([this]() {
      while (read_buffer_.size() < buffer_size_) {
        reached_eof_ = in_->gets(line_c_str_.get(), MAX_LINE_LEN) == nullptr;
        if (reached_eof_) {
          // Reset file ptr to beginning of next file
          in_->close();
          path_idx_ = (path_idx_ + 1) % fnames_.size();

          if (path_idx_ == 0) { reached_eofs_ = true; }

          in_ = getfilehandler(fnames_[path_idx_], read_mode_);
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
    if (reached_eofs_prev_) {
      reached_eofs_prev_ = false;
      return false;
    }

    join_reader();

    reached_eofs_prev_ = reached_eofs_;
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
