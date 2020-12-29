#define CATCH_CONFIG_MAIN  // so that Catch is responsible for main()
#define KOAN_GRAD_CHECKING // Real == double and no lookup approx for sigmoid

#include <catch.hpp>

#include <koan/indexmap.h>
#include <koan/trainer.h>

using namespace koan;

TEST_CASE("Cbow", "[grad]") {
  static_assert(std::is_same<Real, double>::value);

  Table table, ctx;
  unsigned dim = 5;

  IndexMap<std::string> word_map;
  word_map.insert("hello");
  word_map.insert("world");
  word_map.insert("!");
  word_map.insert(".");

  // Prevent trainer from randomly dropping any word.
  std::vector<double> filter_probs{0, 0, 0, 0};

  // Force trainer to sample "." as the negative word.
  std::vector<double> neg_probs{0, 0, 0, 1};

  Sentence sent{0, 1, 2}; // hello world !

  // Randomly initialize center and context word embedding tables.
  for (size_t i = 0; i < word_map.size(); i++) {
    table.push_back(Vector::Random(dim));
    ctx.push_back(Vector::Random(dim));
  }

  Trainer t(
      Trainer::Params{.dim = dim, .ctxs = 5, .negatives = 1, .threads = 1},
      table,
      ctx,
      filter_probs,
      neg_probs);

  // Keep a copy of original weights
  Table table_orig(table), ctx_orig(ctx);

  t.cbow_update(sent,
                /*center*/ 1,
                /*left*/ 0,
                /*right*/ 3,
                /*tid*/ 0,
                /*lr*/ 1,
                /*compute_loss*/ true);

  // analytic gradients
  Table table_agrad(table), ctx_agrad(ctx);
  for (size_t i = 0; i < word_map.size(); i++) {
    table_agrad[i] = table_orig[i] - table[i];
    ctx_agrad[i] = ctx_orig[i] - ctx[i];
  }

  // Compute numeric gradients for every parameter
  Table table_ngrad(table_orig), ctx_ngrad(ctx_orig);

  table = table_orig;
  ctx = ctx_orig;

  // Two-sided numerical gradient:
  // http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
  for (auto tab : {&table, &ctx}) {
    for (size_t i = 0; i < word_map.size(); i++) {
      for (unsigned j = 0; j < dim; j++) {
        const static Real eps = 1e-4;
        Real tmp = tab->at(i)[j];
        tab->at(i)[j] += eps;
        Real loss_up = t.cbow_update(sent,
                                     /*center*/ 1,
                                     /*left*/ 0,
                                     /*right*/ 3,
                                     /*tid*/ 0,
                                     /*lr*/ 1,
                                     /*compute_loss*/ true);
        table = table_orig;
        ctx = ctx_orig;

        tab->at(i)[j] = tmp - eps;
        Real loss_down = t.cbow_update(sent,
                                       /*center*/ 1,
                                       /*left*/ 0,
                                       /*right*/ 3,
                                       /*tid*/ 0,
                                       /*lr*/ 1,
                                       /*compute_loss*/ true);
        table = table_orig;
        ctx = ctx_orig;

        Real num_grad = (loss_up - loss_down) / (2 * eps);
        if (tab == &table) {
          table_ngrad[i][j] = num_grad;
        } else {
          ctx_ngrad[i][j] = num_grad;
        }
      }
    }
  }

  // compare numeric and analytical gradients
  for (size_t i = 0; i < word_map.size(); i++) {
    for (unsigned j = 0; j < dim; j++) {
      CHECK(table_agrad[i][j] == Approx(table_ngrad[i][j]));
      CHECK(ctx_agrad[i][j] == Approx(ctx_ngrad[i][j]));
    }
  }
}

TEST_CASE("Skipgram", "[grad]") {
  static_assert(std::is_same<Real, double>::value);

  Table table, ctx;
  unsigned dim = 5;

  IndexMap<std::string> word_map;
  word_map.insert("hello");
  word_map.insert("world");
  word_map.insert("!");
  word_map.insert(".");

  // Prevent trainer from randomly dropping any word.
  std::vector<double> filter_probs{0, 0, 0, 0};

  // Force trainer to sample "." as the negative word.
  std::vector<double> neg_probs{0, 0, 0, 1};

  Sentence sent{0, 1}; // hello world

  // Randomly initialize center and context word embeddings.
  for (size_t i = 0; i < word_map.size(); i++) {
    table.push_back(Vector::Random(dim));
    ctx.push_back(Vector::Random(dim));
  }

  Trainer t(
      Trainer::Params{.dim = dim, .ctxs = 5, .negatives = 1, .threads = 1},
      table,
      ctx,
      filter_probs,
      neg_probs);

  // Keep a copy of original weights
  Table table_orig(table), ctx_orig(ctx);

  t.sg_update(sent,
              /*center*/ 1,
              /*left*/ 0,
              /*right*/ 2,
              /*tid*/ 0,
              /*lr*/ 1,
              /*compute_loss*/ true);

  // analytic gradients
  Table table_agrad(table), ctx_agrad(ctx);
  for (size_t i = 0; i < word_map.size(); i++) {
    table_agrad[i] = table_orig[i] - table[i];
    ctx_agrad[i] = ctx_orig[i] - ctx[i];
  }

  // Compute numeric gradients for every parameter
  Table table_ngrad(table_orig), ctx_ngrad(ctx_orig);

  table = table_orig;
  ctx = ctx_orig;

  // Two-sided numerical gradient:
  // http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
  for (auto tab : {&table, &ctx}) {
    for (size_t i = 0; i < word_map.size(); i++) {
      for (unsigned j = 0; j < dim; j++) {
        const static Real eps = 1e-4;
        Real tmp = tab->at(i)[j];
        tab->at(i)[j] += eps;
        Real loss_up = t.sg_update(sent,
                                   /*center*/ 1,
                                   /*left*/ 0,
                                   /*right*/ 2,
                                   /*tid*/ 0,
                                   /*lr*/ 1,
                                   /*compute_loss*/ true);
        table = table_orig;
        ctx = ctx_orig;

        tab->at(i)[j] = tmp - eps;
        Real loss_down = t.sg_update(sent,
                                     /*center*/ 1,
                                     /*left*/ 0,
                                     /*right*/ 2,
                                     /*tid*/ 0,
                                     /*lr*/ 1,
                                     /*compute_loss*/ true);
        table = table_orig;
        ctx = ctx_orig;

        Real num_grad = (loss_up - loss_down) / (2 * eps);
        if (tab == &table) {
          table_ngrad[i][j] = num_grad;
        } else {
          ctx_ngrad[i][j] = num_grad;
        }
      }
    }
  }

  // compare numeric and analytical gradients
  for (size_t i = 0; i < word_map.size(); i++) {
    for (unsigned j = 0; j < dim; j++) {
      CHECK(table_agrad[i][j] == Approx(table_ngrad[i][j]));
      CHECK(ctx_agrad[i][j] == Approx(ctx_ngrad[i][j]));
    }
  }
}
