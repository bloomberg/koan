#define CATCH_CONFIG_MAIN // so that Catch is responsible for main()

#include <catch.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <koan/indexmap.h>
#include <koan/sample.h>
#include <koan/trainer.h>

using namespace koan;

/// Draw n samples from sampler and record empirical distribution over classes.
///
/// @param[in] sampler Already-initialized alias sampler
/// @param[in] n       Number of samples to draw
/// @return empirical distribution for sampler
std::vector<Real> sample_dist(AliasSampler sampler, size_t n = 10000000) {
  std::vector<Real> dist(sampler.num_classes(), 0.0);

  for (size_t i = 0; i < n; ++i) { ++dist[sampler.sample()]; }

  for (size_t i = 0; i < dist.size(); ++i) { dist[i] /= n; }

  return dist;
}

/// Test whether the probability of selecting any class from multinomial d2 is
/// within 1% of the probability of selecting it under d1.
///
/// @param[in] d1 First distribution
/// @param[in] d2 Other distribution
/// @return whether distributions are sufficiently close
bool dists_are_close(const std::vector<Real>& d1, const std::vector<Real>& d2) {
  REQUIRE(d1.size() == d2.size());

  for (size_t i = 0; i < d1.size(); ++i) {
    if (std::abs(d1[i] - d2[i]) >= (d1[i] * 0.01)) { return false; }
  }

  return true;
}

TEST_CASE("AliasSampler", "[sample]") {
  std::vector<Real> probs1(2, 0.5);
  AliasSampler sampler1(probs1);

  /// Make sure alias sampler faithfully represents multinomial distributions
  /// where all classes are equally probable.
  SECTION("Balanced binary distribution") {
    CHECK(dists_are_close(probs1, sample_dist(sampler1)));
  }

  std::vector<Real> probs2(10, 0.1);
  AliasSampler sampler2(probs2);

  SECTION("Balanced 10-class") {
    CHECK(dists_are_close(probs2, sample_dist(sampler2)));
  }

  std::vector<Real> probs3(50, 0.02);
  AliasSampler sampler3(probs3);

  SECTION("Balanced 50-class") {
    CHECK(dists_are_close(probs3, sample_dist(sampler3)));
  }

  /// Make sure alias sampler faithfully represents multinomial distributions
  /// where some classes are much more probable than others.
  std::vector<Real> probs4{0.1, 0.9};
  AliasSampler sampler4(probs4);

  SECTION("Unbalanced binary") {
    CHECK(dists_are_close(probs4, sample_dist(sampler4)));
  }

  std::vector<Real> probs5{
      0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.2, 0.2, 0.2, 0.2};
  AliasSampler sampler5(probs5);

  SECTION("Unbalanced 10-class") {
    CHECK(dists_are_close(probs5, sample_dist(sampler5)));
  }
}

TEST_CASE("IndexMap", "[indexmap]") {
  IndexMap<std::string> imap;

  imap.insert("hello");
  imap.insert("world");

  CHECK(imap.size() == 2);
  CHECK(imap.has("hello"));
  CHECK(imap.has("world"));
  CHECK(not imap.has("!"));

  CHECK(imap.lookup("hello") == 0);
  CHECK(imap.lookup("world") == 1);
  CHECK(imap.reverse_lookup(0) == "hello");
  CHECK(imap.reverse_lookup(1) == "world");

  CHECK_THROWS(imap.lookup("!"));
  CHECK_THROWS(imap.reverse_lookup(2));

  SECTION("Insert new") {
    imap.insert("!");

    CHECK(imap.size() == 3);
    CHECK(imap.has("!"));
    CHECK(imap.lookup("!") == 2);
    CHECK(imap.reverse_lookup(2) == "!");
  }

  SECTION("Insert dupe") {
    imap.insert("hello");

    CHECK(imap.size() == 2);
    CHECK(imap.has("hello"));
    CHECK(imap.has("world"));
    CHECK(imap.lookup("hello") == 0);
    CHECK(imap.lookup("world") == 1);
    CHECK(imap.reverse_lookup(0) == "hello");
    CHECK(imap.reverse_lookup(1) == "world");
  }

  SECTION("Clear") {
    imap.clear();

    CHECK(imap.size() == 0);
    CHECK(not imap.has("hello"));
    CHECK(not imap.has("world"));
    CHECK_THROWS(imap.lookup("hello"));
    CHECK_THROWS(imap.lookup("world"));
    CHECK_THROWS(imap.reverse_lookup(0));
    CHECK_THROWS(imap.reverse_lookup(1));
  }
}
