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

#ifndef KOAN_TRAINER_H
#define KOAN_TRAINER_H

#include <random>
#include <vector>

#include "def.h"
#include "sample.h"
#include "sigmoid.h"

namespace koan {

/// Main class to train CBOW and SG word embeddings by negative sampling.
class Trainer {
 public:
  /// Salient parameters of Word2Vec training.
  struct Params {
    unsigned dim = 200;

    // One-sided context extension. E.g. 4 means 4 additional words
    // on each side: [ . . . . x . . . . ]
    unsigned ctxs = 5;

    // Number of negative targets for each positive target for negative
    // sampling
    unsigned negatives = 5;

    // Number of worker threads. This is mainly used to initialize necessary
    // data structures to avoid race conditions. Multithreading itself is
    // done outside of the Trainer class.
    unsigned threads = 8;

    bool use_bad_update = false;
  };

 private:
  // Members
  Params params_;
  // Defines the probability of skipping each word, to downsample highly
  // frequent words
  std::vector<Real> filter_probs_;
  std::vector<Vector> scratch_;                             // one per thread
  std::vector<Vector> scratch2_;                            // one per thread
  std::vector<std::mt19937> gens_;                          // one per thread
  std::vector<std::uniform_real_distribution<Real>> dists_; // one per thread
  std::vector<koan::AliasSampler> neg_samplers_;            // one per thread

  Table& table_; // Input word embeddings (syn1)
  Table& ctx_;   // Output word embeddings (syn0)

 public:
  /// Create trainer
  ///
  /// @param[in] params training parameters
  /// @param[in] table initial input word embeddings (syn1)
  /// @param[in] ctx initial output word embeddings (syn0)
  /// @param[in] filter_probs probability of skipping each word, downsampling
  /// frequent words
  /// @param[in] neg_probs negative sampling probability over vocabulary
  Trainer(Params params,
          Table& table,
          Table& ctx,
          std::vector<Real> filter_probs,
          const std::vector<Real>& neg_probs)
      : params_(params),
        filter_probs_(std::move(filter_probs)),
        scratch_(params_.threads),
        scratch2_(params_.threads),
        neg_samplers_(params_.threads, neg_probs),
        table_(table),
        ctx_(ctx) {
    for (unsigned i = 0; i < params_.threads; i++) {
      gens_.emplace_back(123457 + i);
      dists_.emplace_back(0., 1.);
    }
  }

  // Operations

  /// Update embeddings for a single input sentence, center word, and context
  /// window according to Continuous bag of words (CBOW) objective by negative
  /// sampling.
  ///
  /// @param[in] sent input sentence
  /// @param[in] center_idx index of the center word
  /// @param[in] left index of the leftmost context word (inclusive)
  /// @param[in] right index of the rightmost context word (exclusive)
  /// @param[in] tid thread index
  /// @param[in] lr current learning rate
  /// @param[in] compute_loss whether to also compute and return the CBOW loss.
  /// Used for numerically checking gradient.  If false, will return 0.0
  Real cbow_update(const Sentence& sent,
                   size_t center_idx,
                   size_t left,
                   size_t right,
                   size_t tid,
                   Real lr,
                   bool compute_loss = false) {
    // ISSUE: Neither Mikolov's word2vec nor gensim seems to use the correct
    // gradient which requires normalization by the number of contexts (see
    // below).
    //
    // https://github.com/tmikolov/word2vec/blob/20c129af10659f7c50e86e3be406df663beff438/word2vec.c#L460
    // https://github.com/RaRe-Technologies/gensim/issues/697
    Real loss = 0;
    auto& center_word = ctx_[sent[center_idx]];
    Vector& avg = scratch_[tid];
    Vector& source_idx_grad = scratch2_[tid];
    avg = Vector::Zero(center_word.size());
    source_idx_grad = Vector::Zero(center_word.size());

    // collect embeddings for context words
    static thread_local std::vector<Vector*> sources;
    sources.clear();
    sources.reserve(right - left - 1);

    for (size_t source_idx = left; source_idx < right; source_idx++) {
      if (source_idx != center_idx) {
        auto& v = table_[sent[source_idx]];
        avg += v;
        sources.push_back(&v);
      }
    }

    Real num_source_ids = static_cast<Real>(sources.size());
    if (num_source_ids > 0.) {
      avg /= num_source_ids;

      // Update for positive sample
      // forward pass
      Real sig_pos = sigmoid(avg.dot(center_word));
      if (compute_loss) {
        loss -= std::log(std::max(sig_pos, MIN_SIGMOID_IN_LOSS));
      }
      // backward pass
      if (sig_pos < 1.) {
        if (params_.use_bad_update) {
          // ISSUE above, typical, wrong update!
          source_idx_grad += center_word * ((sig_pos - 1.) * lr);
        } else {
          // ISSUE above, must normalize by number of
          // context words when updating context embeddings
          source_idx_grad +=
              center_word * ((sig_pos - 1.) * lr) / num_source_ids;
        }
        center_word -= avg * ((sig_pos - 1.) * lr);
      }

      // Updates for negative samples
      for (unsigned i = 0; i < params_.negatives; i++) {
        Word random_idx = neg_samplers_[tid].sample();
        if (random_idx == center_idx) { continue; }
        auto& rw = ctx_[random_idx]; // random word
        // forward
        Real sig_neg = sigmoid(avg.dot(rw));
        if (compute_loss) {
          loss -= std::log(std::max(1._R - sig_neg, MIN_SIGMOID_IN_LOSS));
        }
        // backward
        if (sig_neg > 0.) {
          if (params_.use_bad_update) {
            // ISSUE above, typical, wrong update!
            source_idx_grad += rw * (sig_neg * lr);
          } else {
            // ISSUE above
            source_idx_grad += rw * (sig_neg * lr) / num_source_ids;
          }
          rw -= avg * (sig_neg * lr);
        }
      }
      for (auto source : sources) { // update each source (context)
        *source -= source_idx_grad;
      }
    }

    return loss;
  }

  /// Update embeddings for a single input sentence, center word, and context
  /// window according to Skipgram (SG) objective by negative sampling.
  ///
  /// @param[in] sent input sentence
  /// @param[in] center_idx index of the source center word
  /// @param[in] left index of the leftmost context word to predict (inclusive)
  /// @param[in] right index of the rightmost context word to predict
  /// (exclusive)
  /// @param[in] tid thread index
  /// @param[in] lr current learning rate
  /// @param[in] compute_loss whether to also compute and return the SG loss.
  /// Used for numerically checking the gradient.  If false, will return 0.0
  Real sg_update(const Sentence& sent,
                 size_t center_idx,
                 size_t left,
                 size_t right,
                 size_t tid,
                 Real lr,
                 bool compute_loss = false) {
    Real loss = 0;
    auto& center_word = table_.at(sent[center_idx]);
    auto& cw_local = scratch_[tid];
    cw_local = Vector::Zero(center_word.size());

    // Predict each context word given the center
    for (size_t target_idx = left; target_idx < right; target_idx++) {
      if (target_idx != center_idx) {
        auto& target_word = ctx_.at(sent[target_idx]);
        // Update for positive sample
        // forward pass
        Real sig_pos = sigmoid(center_word.dot(target_word));
        if (compute_loss) {
          loss -= std::log(std::max(sig_pos, MIN_SIGMOID_IN_LOSS));
        }
        // backward pass
        if (sig_pos < 1.) {
          cw_local -= target_word * ((sig_pos - 1.) * lr);
          target_word -= center_word * ((sig_pos - 1.) * lr);
        }

        // Update for negative samples
        for (unsigned i = 0; i < params_.negatives; i++) {
          Word random_i = neg_samplers_[tid].sample();
          auto& random_word = ctx_.at(random_i); // random word
          // forward
          Real sig_neg = sigmoid(center_word.dot(random_word));
          if (compute_loss) {
            loss -= std::log(std::max(1 - sig_neg, MIN_SIGMOID_IN_LOSS));
          }
          // backward
          if (sig_neg > 0.) {
            cw_local -= random_word * (sig_neg * lr);
            random_word -= center_word * (sig_neg * lr);
          }
        }
      }
    }
    // cw_local itself is a descent direction, so sign is +=
    center_word += cw_local;
    return loss;
  }

  /// Update embeddings for an entire sentence: treat each word as the center in
  /// turn (modulo downsampled tokens), and sample variable context width.
  ///
  /// @param[in] sent_raw input sentence
  /// @param[in] tid thread index
  /// @param[in] lr learning rate for this instance
  /// @param[in] cbow true if using CBOW loss, else SG
  /// @returns number of tokens in the sentence after downsampling
  size_t train(const Sentence& sent_raw, size_t tid, Real lr, bool cbow) {
    static thread_local Sentence sent(INITIAL_SENTENCE_LEN);
    sent.clear();
    sent.reserve(sent_raw.size());
    for (auto& w : sent_raw) { // prob.at(w) is prob. to discard w
      if (dists_[tid](gens_[tid]) >= filter_probs_.at(w)) { sent.push_back(w); }
    }

    for (size_t center_idx = 0; center_idx < sent.size(); center_idx++) {
      // Sample a contexts width from 1 to maximum context width
      size_t ctxs = 1 + (gens_[tid]() % params_.ctxs);
      size_t left = center_idx > ctxs ? center_idx - ctxs : 0;
      size_t right = std::min(center_idx + ctxs + 1, sent.size());

      if (cbow) { // cbow loss
        cbow_update(sent, center_idx, left, right, tid, lr);
      } else { // skipgram loss
        sg_update(sent, center_idx, left, right, tid, lr);
      }
    }

    return sent.size();
  }
};

} // namespace koan

#endif
