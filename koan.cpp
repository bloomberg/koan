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

#include <atomic>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "extern/mew.h"

#include <koan/cli.h>
#include <koan/def.h>
#include <koan/indexmap.h>
#include <koan/reader.h>
#include <koan/timer.h>
#include <koan/trainer.h>
#include <koan/util.h>

using namespace koan;

int main(int argc, char** argv) {
  srand(123457);
  std::string fname;
  unsigned dim = 200;
  unsigned ctxs = 5;
  unsigned negatives = 5;
  unsigned num_threads = 1;
  unsigned epochs = 1;
  unsigned min_count = 1;
  bool discard = true;
  bool cbow = false;
  Real downsample_th = 1e-3;
  Real init_lr = 0.025; // If cbow, initial learning rate 0.075 recommended.
  Real min_lr = 1e-4;
  Real ns_exponent = 0.75;
  size_t vocab_size = std::numeric_limits<size_t>::max();
  size_t buffer_size = 500'000;
  std::string embedding_path = "";
  bool shuffle = false;
  bool no_progress = false;
  bool partitioned = false;

  std::string pretrained_path;
  std::string continue_vocab = "union";

  Args args;
  args.add(fname, "f,file", "path", "Path to text file", Required);
  args.add(dim, "d,dim", "n", "Word vector dimension");
  args.add(ctxs,
           "c,context-size",
           "n",
           "One sided context size, excluding the center word");
  args.add(negatives,
           "n,negatives",
           "n",
           "Number of negative samples for each positive");
  args.add(
      init_lr,
      "l,learning-rate",
      "x",
      "(Starting) learning rate. 0.025 for skipgram and 0.075 for cbow is recommended.",
      SuggestRange(1e-3, 1e-1));
  args.add(
      min_lr,
      "m,min-learning-rate",
      "x",
      "Minimum (ending) learning rate when linearly scheduling learning rate",
      SuggestRange(0., 1e-4));
  args.add(
      min_count,
      "k,min-count",
      "n",
      "Do not use word identities if raw frequency count is less than n. See --discard");
  args.add(
      discard,
      "i,discard",
      "true|false",
      "If true, discard rare words (see --min-count) else, convert them to UNK");
  args.add(cbow,
           "b,cbow",
           "true|false",
           "If true, use cbow loss instead of skipgram");
  args.add(
      downsample_th, "o,downsample-threshold", "x", "Downsample threshold");
  args.add(ns_exponent,
           "x,ns-exponent",
           "x",
           "Exponent for negative sampling distribution",
           RequireRange(0., 1.));
  args.add(epochs, "e,epochs", "n", "Training epochs");
  args.add(vocab_size,
           "V,vocab-size",
           "n",
           "Vocabulary size to pick top n words instead of all");
  args.add(num_threads, "t,threads", "n", "Number of worker threads");
  args.add(buffer_size,
           "B,buffer-size",
           "n",
           "Buffer size in number of sentences. Memory footprint is in the "
           "order of buffer-size × avg. length of sentence. Larger buffer-size "
           "is bigger memory footprint but better shuffling.");
  args.add(
      embedding_path,
      "p,embedding-path",
      "path",
      "Path embeddings should be saved to.  Defaults to saving to a file named 'embeddings_${CURRENT_DATETIME}.txt'");
  args.add(
      pretrained_path,
      "r,pretrained-path",
      "path",
      "If passed (nonempty), continue training from an existing embedding table (also see continue-vocab)");
  args.add(
      continue_vocab,
      "v,continue-vocab",
      "old|new|union",
      "Which vocab to use when continuing training (see pretrained-path), old: from pretrained table, new: from data, union: combined",
      RequireFromSet({"old", "new", "union"}));
  args.add(
      shuffle,
      "s,shuffle-sentences",
      "true|false",
      "If true, will shuffle sentences in a batch before allocating to worker threads rather than assigning them consecutively to threads");
  args.add(
      partitioned,
      "L,partitioned",
      "true|false",
      "If true, use the partitioned version of main parallel for loop. Can be "
      "faster due to a lack of std::atomic use, but also slower due to workers "
      "with less work waiting for others. Changes sentence processing order.");
  args.add_flag(no_progress,
                "P,no-progress",
                "If passed, do not display counters and progress bars.");

  args.add_help();
  args.parse(argc, argv);

  Table table, ctx, local(num_threads, Vector::Zero(dim));
  std::vector<std::string> ordered_vocab;
  IndexMap<std::string_view> word_map; // ordered_vocab will own the
                                       // actual strings.

  std::unordered_map<std::string, Vector> pretrained_table;

  if (not pretrained_path.empty()) {
    long unsigned lines = 0;
    auto counter = mew::Counter(lines,
                                "Reading pretrained embeddings",
                                "lines/s",
                                mew::Speed::Last,
                                1.);
    if (no_progress) {
      std::cout << "Reading pretrained embeddings..." << std::endl;
    } else {
      counter.start();
    }

    std::vector<std::string> s;
    s.reserve(100);

    readlines(pretrained_path, [&](const std::string_view& line) {
      s.clear();
      split(s, line, ' ');
      KOAN_ASSERT(dim == (s.size() - 1),
                  "Specified dimension doesn't match pretrained table!");
      auto& word = s[0];
      if (pretrained_table.find(word) != pretrained_table.end()) {
        std::cerr << "WARNING: Pretrained table has duplicate entry: \"" << word
                  << "\". Skipping.\n";
      } else {
        Vector v(dim);
        for (Vector::Index i = 0; i < v.size(); i++) {
          v[i] = std::stof(s[i + 1]);
        }
        pretrained_table.emplace(word, std::move(v));
      }
      lines++;
    });

    counter.done();
  }

  if (not discard) { ordered_vocab.push_back(UNKSTR); }

  if (continue_vocab == "old" or continue_vocab == "union") {
    for (auto& pair : pretrained_table) { ordered_vocab.push_back(pair.first); }
  }

  long unsigned tot = 0;
  long unsigned lines = 0;
  bool read_whole_data = false;

  std::unordered_map<std::string, unsigned long long> freqs;
  freqs.reserve(INITIAL_INDEX_SIZE);

  {
    auto counter =
        mew::Counter(lines, "Building vocab", "lines/s", mew::Speed::Last, 1.);
    if (no_progress) {
      std::cout << "Building vocab..." << std::endl;
    } else {
      counter.start();
    }

    Timer t;
    std::vector<std::string> s;
    s.reserve(100);
    readlines(fname, [&](const std::string_view& line) {
      s.clear();
      split(s, line, ' ');
      for (auto& w : s) { freqs[w]++; }
      lines++;
      tot += s.size();
    });

    for (auto& [word, count] : freqs) {
      bool add_word = continue_vocab != "old" and count >= min_count;
      if (continue_vocab == "union") {
        add_word =
            add_word and pretrained_table.find(word) == pretrained_table.end();
      }
      if (add_word) { ordered_vocab.push_back(word); }
    }

    for (auto& p : pretrained_table) {
      auto& word = p.first;
      freqs.emplace(word, min_count);
    }

    size_t begin_offset = discard ? 0 : 1; // keep UNK at 0 if exists
    std::sort(ordered_vocab.begin() + begin_offset,
              ordered_vocab.end(),
              [&](auto& a, auto& b) { return freqs[a] > freqs[b]; });

    // Resize if vocab is bigger than specified size
    if (vocab_size < ordered_vocab.size()) { ordered_vocab.resize(vocab_size); }
    if (not no_progress) { counter.done(); }
    std::cout << "Done in " << unsigned(t.s()) << "s." << std::endl;

    KOAN_ASSERT(ordered_vocab.size() < std::numeric_limits<Word>::max(),
                "Vocab is too big for Word type! Either shrink vocab, or use "
                "bigger Word type.");

    for (const auto& w : ordered_vocab) {
      word_map.insert(std::string_view(w));
      assert(word_map.lookup(w) == table.size());
      assert(word_map.lookup(w) == ctx.size());
      table.push_back(Vector::Zero(dim));
      ctx.push_back(Vector::Zero(dim));
    }

    if (buffer_size > lines) {
      std::cerr << "WARNING: Buffer size is larger than the total number"
                   " of sentences in the corpus -- will load entire dataset"
                   " into memory once instead of streaming.\n";
      read_whole_data = true;
    }
  }

  std::vector<Real> prob(ordered_vocab.size());     // filter probs
  std::vector<Real> neg_prob(ordered_vocab.size()); // neg sampling probs

  if (not discard) { freqs[UNKSTR] = 0; }
  for (Word w = 0; w < prob.size(); w++) {
    prob[w] = neg_prob[w] = freqs.at(std::string(word_map.reverse_lookup(w)));
  }

  // Maybe filter words by frequency
  // -
  // https://github.com/svn2github/word2vec/blob/99e546e27cae10aa20209dae1ed98716ac9022e9/word2vec.c#L396
  // -
  // https://github.com/RaRe-Technologies/gensim/blob/e859c11f6f57bf3c883a718a9ab7067ac0c2d4cf/gensim/models/word2vec.py#L1536
  for (auto& p : prob) {
    p = p / tot;
    p = 1. - sqrt(downsample_th / p) -
        downsample_th / p; // probability of discarding
  }

  // Compute negative sampling probs
  // https://github.com/RaRe-Technologies/gensim/blob/e859c11f6f57bf3c883a718a9ab7067ac0c2d4cf/gensim/models/word2vec.py#L1608
  {
    std::transform(neg_prob.begin(),
                   neg_prob.end(),
                   neg_prob.begin(),
                   [ns_exponent](auto& x) { return std::pow(x, ns_exponent); });
    Real total = std::accumulate(neg_prob.begin(), neg_prob.end(), 0.);
    std::transform(neg_prob.begin(),
                   neg_prob.end(),
                   neg_prob.begin(),
                   [total](auto& x) { return x / total; });
  }

  // Randomly initialize embeddings for words not present in pretrained_table
  for (size_t w = 0; w < table.size(); w++) {
    std::string word(word_map.reverse_lookup(w));
    if (pretrained_table.find(word) != pretrained_table.end()) {
      table[w] = std::move(pretrained_table[word]);
    } else {
      table[w].setRandom();
      table[w] *= (0.5 / dim);
    }
    ctx[w].setZero();
  }
  // pretrained_table not needed after here, save memory
  pretrained_table.clear();

  Trainer::Params params{
      .dim = dim,
      .ctxs = ctxs,
      .negatives = negatives,
      .threads = num_threads,
  };

  Trainer trainer(params, table, ctx, prob, neg_prob);
  std::mt19937 g(12345);

  std::atomic<size_t> tokens{0}, sents{0}, total_tokens{0};

  Sentences sentences;

  Timer t;
  std::unique_ptr<Reader> reader;
  if (read_whole_data) {
    reader = std::make_unique<OnceReader>(word_map, fname, discard);
  } else {
    reader =
        std::make_unique<AsyncReader>(word_map, fname, buffer_size, discard);
  }
  for (size_t e = 0; e < epochs; e++) {
    std::atomic<size_t> filtered_tokens_in_epoch{0}, total_tokens_in_epoch{0};

    tokens = 0;
    sents = 0;

    std::cout << "Epoch " << e << std::endl;

    auto bar = mew::ProgressBar(sents, lines, "Sents:") |
               mew::Counter(tokens, "Toks:", "tok/s", mew::Speed::Last);
    if (not no_progress) { bar.start(); }

    while (reader->get_next(sentences)) {
      std::vector<size_t> perm(sentences.size());
      std::iota(perm.begin(), perm.end(), 0);

      if (shuffle) { std::shuffle(perm.begin(), perm.end(), g); }

      auto work = [&](size_t i, size_t tid) {
        auto& s = sentences[perm[i]];

        // linear learning rate scheduling
        // https://github.com/RaRe-Technologies/gensim/blob/374de281b27f21fac4df20c315ee07caafb279c0/gensim/models/base_any2vec.py#L1083
        Real lr_sched =
            Real(e) / epochs + (Real(i) / sentences.size()) / epochs;
        Real lr = init_lr - (init_lr - min_lr) * lr_sched;

        size_t remaining_toks = trainer.train(s, tid, lr, cbow);
        sents++;
        tokens += remaining_toks;
        total_tokens += remaining_toks;
        filtered_tokens_in_epoch += remaining_toks;
        total_tokens_in_epoch += s.size();
      };

      if (partitioned) {
        parallel_for_partitioned(0, sentences.size(), work, num_threads);
      } else {
        parallel_for(0, sentences.size(), work, num_threads);
      }
    }

    if (not no_progress) { bar.done(); }

    std::cout << std::fixed << std::setprecision(2)
              << 100. * filtered_tokens_in_epoch / total_tokens_in_epoch
              << "% of tokens were retained while filtering." << std::endl;
  }
  auto total_secs = t.s();
  std::cout << "Took " << unsigned(total_secs) << "s. (excluding vocab build)"
            << std::endl
            << "Overall speed was " << total_tokens / total_secs << " toks/s"
            << std::endl;

  if (embedding_path.empty()) {
    embedding_path = "embeddings_" + date_time("%F_%T") + ".txt";
  }

  {
    std::cout << "Saving to " << embedding_path << std::endl;
    FILE* out = fopen(embedding_path.c_str(), "w");
    KOAN_ASSERT(out);
    std::string buf;
    buf.reserve(MAX_LINE_LEN);
    for (auto& w : word_map.keys()) {
      buf.clear();
      buf += w;
      auto v = table[word_map.lookup(w)];
      for (int j = 0; j < v.size(); j++) {
        buf += " ";
        buf += std::to_string(v(j));
      }
      buf += "\n";
      fputs(buf.data(), out);
    }
    fclose(out);
  }
}
