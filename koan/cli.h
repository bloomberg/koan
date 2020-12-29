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

// Tiny command line parsing utilities

#ifndef KOAN_CLI_H
#define KOAN_CLI_H

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "extern/tblr.h"

#include "util.h"

namespace koan {
namespace internal {

namespace fromstr { // Convert from string to things

template <typename T>
inline T to(const std::string& s);

template <>
inline std::string to<std::string>(const std::string& s) {
  return s;
}

template <>
inline float to<float>(const std::string& s) {
  return std::stof(s);
}

template <>
inline double to<double>(const std::string& s) {
  return std::stod(s);
}

template <>
inline unsigned to<unsigned>(const std::string& s) {
  return std::stoul(s);
}

template <>
inline int to<int>(const std::string& s) {
  return std::stoi(s);
}

template <>
inline long to<long>(const std::string& s) {
  return std::stol(s);
}

template <>
inline unsigned long to<unsigned long>(const std::string& s) {
  return std::stoul(s);
}

template <>
inline bool to<bool>(const std::string& s) {
  if (s == "true" or s == "True" or s == "1") { return true; }
  if (s == "false" or s == "False" or s == "0") { return false; }
  throw std::runtime_error("Unexpected boolean string: " + s);
  return false;
}

} // namespace fromstr

template <typename T>
std::string tostr(const T& x) {
  std::ostringstream ss;
  ss << x;
  return ss.str();
}

} // namespace internal

enum Require { Optional, Required };

class Args;

namespace internal {

class ValidityBase {
 protected:
  const bool throw_;
  const std::string namestr_;

 public:
  ValidityBase(bool throws, std::string namestr)
      : throw_(throws), namestr_(std::move(namestr)) {}
  virtual ~ValidityBase() = default;

  virtual std::string str() const = 0;

  friend class koan::Args;
};

class ArgBase {
 public:
  virtual ~ArgBase() {}
  ArgBase(std::string descr,
          std::vector<std::string> names,
          std::string value,
          Require required,
          std::unique_ptr<ValidityBase> validity = nullptr)
      : descr_(std::move(descr)),
        names_(std::move(names)),
        value_(std::move(value)),
        required_(required),
        validity_(std::move(validity)) {}
  virtual std::string value_str() const = 0;
  virtual bool is_flag() const = 0;

 protected:
  std::string descr_;              // description of the option in the helpstr
  std::vector<std::string> names_; // names of the option
  std::string value_;              // value name in the helpstr
  Require required_ = Optional;    // Required vs Optional
  bool parsed_ = false;            // is this already parsed
  std::unique_ptr<ValidityBase> validity_; // checks if parsed value is valid

  void parse(const std::string& value) {
    parse_(value);
    parsed_ = true;
  }

 private:
  virtual void parse_(const std::string& value) = 0;

  friend class koan::Args;
};

template <typename T>
class Validity : public ValidityBase {
 public:
  Validity(bool throws, std::string namestr)
      : ValidityBase(throws, std::move(namestr)) {}

  virtual void check(const T&) const {
    throw std::runtime_error("A validity checker is not implemented for "
                             "this type!");
  }

  std::string str() const override { return ""; }

  void warn_or_throw(const T& val, bool is_range) const {
    std::string adj = throw_ ? "required" : "suggested";
    std::string noun = is_range ? "range" : "set";
    std::string msg = "Value " + tostr(val) + " for " + namestr_ +
                      " is not in " + adj + " " + noun + ": " + str();
    if (throw_) {
      throw std::runtime_error(msg);
    } else {
      std::cerr << "Warning: " + msg << std::endl;
    }
  }
};

// Check if parsed argument value is in the set
template <typename T>
class InSet : public Validity<T> {
 private:
  const std::vector<T> candidates_;

 public:
  InSet(bool throws, std::vector<T> candidates, std::string namestr)
      : Validity<T>(throws, std::move(namestr)),
        candidates_(std::move(candidates)) {}
  void check(const T& val) const override {
    if (std::find(candidates_.begin(), candidates_.end(), val) ==
        candidates_.end()) {
      this->warn_or_throw(val, false);
    }
  }
  std::string str() const override {
    std::string s = "{";
    size_t i = 0;
    while (i < candidates_.size()) {
      std::string value_str = tostr(candidates_[i]);
      i++;
      if (value_str.size() + s.size() > 20) { break; }
      if (i > 1) { s += ", "; }
      s += value_str;
    }
    if (i == candidates_.size()) {
      s += "}";
    } else {
      s += ", ...}";
    }
    return s;
  }
};

// Check if parsed argument value is in the range (inclusive)
template <typename T>
class InRange : public Validity<T> {
 private:
  const T left_, right_;

 public:
  InRange(bool throws, const T& left, const T& right, std::string namestr)
      : Validity<T>(throws, std::move(namestr)), left_(left), right_(right) {}
  void check(const T& val) const override {
    if (not(left_ <= val and val <= right_)) { this->warn_or_throw(val, true); }
  }
  std::string str() const override {
    return "[" + tostr(left_) + ", " + tostr(right_) + "]";
  }
};

template <typename T>
class Arg : public ArgBase {
 protected:
  T& dest_;

 public:
  Arg<T>(T& dest,
         std::string descr,
         std::vector<std::string> names,
         std::string value,
         Require required = Optional,
         std::unique_ptr<ValidityBase> validity = nullptr)
      : ArgBase(std::move(descr),
                std::move(names),
                std::move(value),
                required,
                std::move(validity)),
        dest_(dest) {}

  bool is_flag() const override { return false; }

  std::string value_str() const override { return tostr(dest_); }

 private:
  void parse_(const std::string& value) override {
    dest_ = internal::fromstr::to<T>(value);
    if (validity_) { dynamic_cast<Validity<T>&>(*validity_).check(dest_); }
  }
};

template <>
class Arg<bool> : public ArgBase {
 protected:
  bool& dest_;
  bool is_flag_ = false;

 public:
  Arg<bool>(bool& dest,
            std::string descr,
            std::vector<std::string> names,
            std::string value,
            Require required = Optional,
            std::unique_ptr<Validity<bool>> /*validity*/ = nullptr,
            bool is_flag = false)
      : ArgBase(std::move(descr), std::move(names), std::move(value), required),
        dest_(dest),
        is_flag_(is_flag) {}

  bool is_flag() const override { return is_flag_; }

  std::string value_str() const override { return (dest_ ? "true" : "false"); }

 private:
  void parse_(const std::string& value) override {
    if (is_flag_) {
      dest_ = true;
    } else {
      dest_ = internal::fromstr::to<bool>(value);
    }
  }
};

template <typename T>
class Arg<std::function<T(void)>> : public ArgBase {
  // Functional args are flags that are used to perform actions
 protected:
  std::function<T(void)> f_;

 public:
  Arg<std::function<T(void)>>(std::function<T(void)> f,
                              std::string descr,
                              std::vector<std::string> names,
                              std::string value)
      : ArgBase(std::move(descr), std::move(names), std::move(value), Optional),
        f_(f) {}
  bool is_flag() const override { return true; }
  std::string value_str() const override { return ""; }

 private:
  void parse_(const std::string&) override { f_(); }
};

template <typename T>
struct Range {
  bool throws;
  T left, right;
};

template <typename T>
struct Set {
  bool throws;
  std::vector<T> candidates;
};

} // namespace internal

template <typename T>
auto RequireRange(const T& left, const T& right) {
  return internal::Range<T>{/*throws*/ true, left, right};
}

template <typename T>
auto SuggestRange(const T& left, const T& right) {
  return internal::Range<T>{/*throws*/ false, left, right};
}

template <typename T>
auto RequireFromSet(const std::vector<T>& candidates) {
  return internal::Set<T>{/*throws*/ true, candidates};
}

template <typename T>
auto SuggestFromSet(const std::vector<T>& candidates) {
  return internal::Set<T>{/*throws*/ false, candidates};
}

template <typename T>
auto RequireFromSet(std::initializer_list<T> candidates) {
  return internal::Set<T>{/*throws*/ true, candidates};
}

template <typename T>
auto SuggestFromSet(std::initializer_list<T> candidates) {
  return internal::Set<T>{/*throws*/ false, candidates};
}

class Args {
 public:
  struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

 private:
  using ArgBase = internal::ArgBase;
  template <typename T>
  using Arg = internal::Arg<T>;
  template <typename T>
  using Validity = internal::Validity<T>;

  std::vector<std::unique_ptr<ArgBase>> positional_args_;
  std::vector<std::unique_ptr<ArgBase>> named_args_;
  std::map<std::string, size_t> name2i_;
  bool has_help_ = false;
  std::string program_name_ = "program";

  static bool is_name(const std::string& value, std::string& name);
  static std::vector<std::string> validate_names(const std::string& namestr);
  static void ensure(bool predicate, const std::string& msg);
  std::string helpstr() const;

 public:
  void parse(int argc, char** argv);
  void parse(const std::vector<std::string>& argv);

  // add option (named arg)
  template <typename T>
  void add(T& dest,
           const std::string& namestr,
           const std::string& value,
           const std::string& descr,
           Require require = Optional);

  template <typename T, typename T2>
  void add(T& dest,
           const std::string& namestr,
           const std::string& value,
           const std::string& descr,
           const internal::Range<T2>& range,
           Require require = Optional);

  template <typename T, typename T2>
  void add(T& dest,
           const std::string& namestr,
           const std::string& value,
           const std::string& descr,
           const internal::Set<T2>& range,
           Require require = Optional);

  // add positional arg
  template <typename T>
  void add(T& dest, const std::string& value, const std::string& descr);

  // add option as a flag
  void add_flag(bool& dest,
                const std::string& namestr,
                const std::string& descr,
                Require require = Optional);

  // add helpstring flag (-?, -h, --help)
  void add_help();
};

inline void Args::ensure(bool predicate, const std::string& msg) {
  if (not predicate) { throw ParseError(msg.c_str()); }
}

inline bool Args::is_name(const std::string& value, std::string& name) {
  if (value.size() >= 2 and value[0] == '-' and value[1] == '-') {
    name = value.substr(2, value.size());
    ensure(not name.empty(), "Prefix `--` not followed by an option!");
    return true;
  }
  if (value.size() >= 1 and value[0] == '-') {
    name = value.substr(1, value.size());
    ensure(not name.empty(), "Prefix `-` not followed by an option!");
    ensure(name.size() == 1,
           "Options prefixed by `-` have to be short names! "
           "Did you mean `--" +
               name + "`?");
    return true;
  }
  return false;
}

inline std::string join(const std::vector<std::string>& strings,
                        const std::string& delim) {
  std::string s;
  for (size_t i = 0; i < strings.size(); i++) {
    if (i > 0) { s += delim; }
    s += strings[i];
  }
  return s;
}

inline void Args::parse(const std::vector<std::string>& argv) {
  size_t i = 0;
  size_t positional_i = 0;

  while (i < argv.size()) {
    std::string name;
    if (is_name(argv[i], name)) { // a named argument (option)
      ensure(name2i_.find(name) != name2i_.end(), "Unexpected option: " + name);
      auto& opt = *named_args_.at(name2i_.at(name));
      ensure(not opt.parsed_, "Option `" + name + "` is multiply given!");
      if (opt.is_flag()) {
        opt.parse("");
        i++;
      } else {
        ensure((i + 1) < argv.size(),
               "Option `" + name + "` is missing value!");
        opt.parse(argv.at(i + 1));
        i += 2;
      }
    } else { // a positional argument
      ensure(positional_i < positional_args_.size(),
             "Unexpected positional argument: " + argv[i]);
      positional_args_.at(positional_i)->parse(argv[i]);
      i++;
      positional_i++;
    }
  }

  // check if all required args are parsed
  for (auto& arg : positional_args_) {
    ensure(arg->parsed_,
           "Required positional argument <" + arg->value_ +
               "> is not provided!");
  }
  for (auto& arg : named_args_) {
    if (arg->required_) {
      ensure(arg->parsed_,
             "Required option `" + join(arg->names_, ", ") + " <" +
                 arg->value_ + ">` is not provided!");
    }
  }
}

inline void Args::parse(int argc, char** argv) {
  program_name_ = argv[0];
  std::vector<std::string> argv_;
  for (int i = 1; i < argc; i++) { argv_.push_back(argv[i]); }
  parse(argv_);
}

inline void Args::add_help() {
  const static std::vector<std::string> names({"?", "h", "help"});
  for (auto& name : names) {
    ensure(name2i_.find(name) == name2i_.end(),
           "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }
  named_args_.push_back(std::make_unique<Arg<std::function<void(void)>>>(
      [this]() {
        std::cout << helpstr() << std::flush;
        exit(0);
      },
      "print this help message and quit",
      names,
      ""));
  has_help_ = true;
}

inline std::vector<std::string>
Args::validate_names(const std::string& namestr) {
  auto names = split<std::string>(namestr, ',');
  ensure(not names.empty(), "Option name cannot be empty!");
  ensure(names.size() <= 2,
         "Option names can be one short and one long at most! "
         "E.g. \"o,option\" or \"o\" or \"option\".");
  if (names.size() == 2) { // ether specify "o,option",
    auto& short_name = names[0];
    auto& long_name = names[1];
    ensure(short_name.size() == 1 and long_name.size() > 1,
           "Multiple form option names should be first short then long! "
           "E.g. \"o,option\".");
  } else { // or "o" only or "option" only.
    ;
  }

  return names;
}

template <typename T>
inline void Args::add(T& dest,
                      const std::string& namesstr,
                      const std::string& value,
                      const std::string& descr,
                      Require required) {
  auto names = validate_names(namesstr);
  for (auto& name : names) {
    ensure(name2i_.find(name) == name2i_.end(),
           "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }
  named_args_.push_back(
      std::make_unique<Arg<T>>(dest, descr, names, value, required));
  // I cannot use make_unique because I made the ctor protected. Is this OK?
}

template <typename T, typename T2>
inline void Args::add(T& dest,
                      const std::string& namesstr,
                      const std::string& value,
                      const std::string& descr,
                      const internal::Range<T2>& range,
                      Require required) {
  auto names = validate_names(namesstr);
  for (auto& name : names) {
    ensure(name2i_.find(name) == name2i_.end(),
           "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }
  named_args_.push_back(std::make_unique<Arg<T>>(
      dest,
      descr,
      names,
      value,
      required,
      std::make_unique<internal::InRange<T>>(
          range.throws, range.left, range.right, namesstr)));
}

template <typename T, typename T2>
inline void Args::add(T& dest,
                      const std::string& namesstr,
                      const std::string& value,
                      const std::string& descr,
                      const internal::Set<T2>& set,
                      Require required) {
  auto names = validate_names(namesstr);
  for (auto& name : names) {
    ensure(name2i_.find(name) == name2i_.end(),
           "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }
  std::vector<T> candidate_set;
  for (auto& val : set.candidates) { candidate_set.push_back(val); }
  named_args_.push_back(
      std::make_unique<Arg<T>>(dest,
                               descr,
                               names,
                               value,
                               required,
                               std::make_unique<internal::InSet<T>>(
                                   set.throws, candidate_set, namesstr)));
}

template <typename T>
inline void
Args::add(T& dest, const std::string& value, const std::string& descr) {
  positional_args_.emplace_back(new Arg<T>(dest, descr, {}, value, Required));
  // I cannot use make_unique because I made the ctor protected. Is this OK?
}

inline void Args::add_flag(bool& dest,
                           const std::string& namestr,
                           const std::string& descr,
                           Require require) {
  ensure(not dest,
         "Optional boolean flags need to default to false, "
         "since the action is `store true`!");
  auto names = validate_names(namestr);
  for (auto& name : names) {
    ensure(name2i_.find(name) == name2i_.end(),
           "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }
  named_args_.emplace_back(
      new Arg<bool>(dest, descr, names, "", require, nullptr, true));
}

inline std::string Args::helpstr() const {
  auto table = []() {
    tblr::Table t;
    t.widths({0, 50}).multiline(tblr::Space).layout(tblr::indented_list());
    return t;
  };

  std::stringstream ss;

  ss << "Usage:\n  " << program_name_;
  for (auto& arg_ : positional_args_) { ss << " <" << arg_->value_ << ">"; }
  ss << " options\n";

  if (not positional_args_.empty() or not named_args_.empty()) {
    ss << "\nwhere ";
  }

  if (not positional_args_.empty()) {
    auto t = table();
    for (auto& arg_ : positional_args_) {
      t << (tblr::Cell() << "<" << arg_->value_ << ">") << arg_->descr_
        << tblr::endr;
    }
    ss << "positional arguments are:\n" << t << "\n";
  }

  std::vector<size_t> required_opts, optional_opts;
  for (size_t i = 0; i < named_args_.size(); i++) {
    if (named_args_[i]->required_) {
      required_opts.push_back(i);
    } else {
      optional_opts.push_back(i);
    }
  }

  auto make_option_str = [&](auto& arg) {
    auto names = arg->names_;
    for (auto& name : names) {
      if (name.size() == 1) {
        name = "-" + name;
      } else {
        name = "--" + name;
      }
    }
    return join(names, ", ");
  };

  if (not required_opts.empty()) {
    auto t = table();
    for (size_t i = 0; i < required_opts.size(); i++) {
      auto& arg_ = named_args_.at(required_opts[i]);

      auto option = (tblr::Cell() << make_option_str(arg_));
      if (not arg_->is_flag()) { option << " <" << arg_->value_ << ">"; }
      auto descr = (tblr::Cell() << arg_->descr_);

      if (arg_->validity_) {
        if (arg_->validity_->throw_) {
          descr << " (required in ";
        } else {
          descr << " (suggested in ";
        }
        descr << arg_->validity_->str();
        descr << ")";
      }

      t << option << descr << tblr::endr;
    }

    ss << "required options are:\n" << t << "\n";
  }

  if (not optional_opts.empty() or has_help_) {
    auto t = table();
    if (has_help_) {
      t << "-?, -h, --help" << named_args_[name2i_.at("h")]->descr_
        << tblr::endr;
    }

    for (size_t i = 0; i < optional_opts.size(); i++) {
      auto& arg_ = named_args_.at(optional_opts[i]);

      if (has_help_ and &named_args_[name2i_.at("h")] == &arg_) { continue; }

      auto option = (tblr::Cell() << make_option_str(arg_));
      if (not arg_->is_flag()) { option << " <" << arg_->value_ << ">"; }
      auto descr = (tblr::Cell() << arg_->descr_);
      if (arg_->is_flag()) {
        descr << " (flag)";
      } else {
        descr << " (default: " << arg_->value_str();
        if (arg_->validity_) {
          if (arg_->validity_->throw_) {
            descr << ", required in ";
          } else {
            descr << ", suggested in ";
          }
          descr << arg_->validity_->str();
        }
        descr << ")";
      }

      t << option << descr << tblr::endr;
    }
    ss << "optional options are:\n" << t << "\n";
  }
  return ss.str();
}

} // namespace koan

#endif
