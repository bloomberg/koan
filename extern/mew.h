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

#ifndef MEW_H
#define MEW_H

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace mew {

using Strings = std::vector<std::string>;
using StringsList = std::vector<Strings>;

// double precision seconds
using Duration = std::chrono::duration<double, std::ratio<1>>;

enum AnimationStyle : unsigned short {
  Ellipsis,
  Clock,
  Moon,
  Earth,
  Bar,
  Square,
};

const static StringsList animation_stills_{
    {".  ", ".. ", "..."},
    {"🕐", "🕜", "🕑", "🕝", "🕒", "🕞", "🕓", "🕟", "🕔", "🕠", "🕕", "🕡",
     "🕖", "🕢", "🕗", "🕣", "🕘", "🕤", "🕙", "🕥", "🕚", "🕦", "🕛", "🕧"},
    {"🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔"},
    {"🌎", "🌍", "🌏"},
    {"-", "/", "|", "\\"},
    {"▖", "▘", "▝", "▗"},
};

enum ProgressBarStyle : unsigned short { Bars, Blocks, Arrow };

const static StringsList progress_partials_{
    {"|"},
    {"▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"},
    {">", "="},
};

enum class Speed : unsigned short { None, Last, Overall, Both };

template <typename Derived>
class AsyncDisplay {
 private:
  Duration period_;
  std::unique_ptr<std::thread> displayer_;
  std::condition_variable completion_;
  std::mutex completion_m_;
  bool complete_ = false;

  std::string message_;
  std::ostream& out_;

  // Render animation, progress bar, etc. Needs to be specialized.
  void render_(std::ostream& out) { static_cast<Derived&>(*this).render_(out); }

  // Display message (maybe with animation, progress bar, etc)
  void display_() {
    out_ << "\r";
    render_(out_);
    out_ << std::flush;
  }

 protected:
  void render_message_(std::ostream& out) {
    if (not message_.empty()) { out << message_ << " "; }
  }

 public:
  AsyncDisplay(std::string message = "",
               double period = 1,
               std::ostream& out = std::cout)
      : period_(period), message_(std::move(message)), out_(out) {}

  AsyncDisplay(std::string message,
               Duration period,
               std::ostream& out = std::cout)
      : period_(period), message_(std::move(message)), out_(out) {}

  AsyncDisplay(const AsyncDisplay& other)
      : period_(other.period_),
        complete_(other.complete_),
        message_(other.message_),
        out_(other.out_) {}

  AsyncDisplay(AsyncDisplay&& other)
      : period_(std::move(other.period_)),
        complete_(std::move(other.complete_)),
        message_(std::move(other.message_)),
        out_(other.out_) {}

  void start() {
    displayer_ = std::make_unique<std::thread>([this]() {
      display_();
      while (true) {
        std::unique_lock<std::mutex> lock(completion_m_);
        completion_.wait_for(lock, period_);
        display_();
        if (complete_) { break; }
      }
    });
  }

  void done() {
    if (not displayer_) { return; } // already done() before; noop
    {
      std::lock_guard<std::mutex> lock(completion_m_);
      complete_ = true;
    }
    completion_.notify_all();
    displayer_->join();
    displayer_.reset();
    out_ << std::endl;
  }

  template <typename Left, typename Right>
  friend class Composite;
};

class Animation : public AsyncDisplay<Animation> {
 private:
  unsigned short frame_ = 0;
  const Strings& stills_;

  void render_(std::ostream& out) {
    this->render_message_(out);
    out << stills_[frame_] << " ";
    frame_ = (frame_ + 1) % stills_.size();
  }

 public:
  using Style = AnimationStyle;

  Animation(std::string message = "",
            Style style = Ellipsis,
            double period = 1,
            std::ostream& out = std::cout)
      : AsyncDisplay<Animation>(message, period, out),
        stills_(animation_stills_[static_cast<unsigned short>(style)]) {}

  Animation(const Animation&) = default;
  Animation(Animation&&) = default;

  friend class AsyncDisplay<Animation>;
  template <typename Left, typename Right>
  friend class Composite;
};

template <typename LeftDisplay, typename RightDisplay>
class Composite : public AsyncDisplay<Composite<LeftDisplay, RightDisplay>> {
 private:
  LeftDisplay left_;
  RightDisplay right_;

  void render_(std::ostream& out) {
    left_.render_(out);
    out << " ";
    right_.render_(out);
  }

 public:
  Composite(LeftDisplay left, RightDisplay right)
      : AsyncDisplay<Composite<LeftDisplay, RightDisplay>>(left.message_,
                                                           left.period_,
                                                           left.out_),
        left_(std::move(left)),
        right_(std::move(right)) {}

  friend class AsyncDisplay<Composite<LeftDisplay, RightDisplay>>;
  template <typename Left, typename Right>
  friend class Composite;
};

template <typename LeftDisplay, typename RightDisplay>
auto operator|(LeftDisplay left, RightDisplay right) {
  return Composite<LeftDisplay, RightDisplay>(std::move(left),
                                              std::move(right));
}

template <typename Progress>
struct ProgressTraits {
  using value_type = Progress;
};

template <typename Progress>
struct ProgressTraits<std::atomic<Progress>> {
  using value_type = Progress;
};

template <typename Progress>
class Speedometer {
 private:
  Progress& progress_;        // Current amount of work done
  Speed speed_;               // Time period to compute speed over
  std::string unit_of_speed_; // unit (message) to display alongside speed

  using Clock = std::chrono::system_clock;
  using Time = std::chrono::time_point<Clock>;

  Time start_time_, last_start_time_;
  typename ProgressTraits<Progress>::value_type last_progress_{0};

 public:
  void render_speed(std::ostream& out) {
    if (speed_ != Speed::None) {
      std::stringstream ss; // use local stream to avoid disturbing `out` with
                            // std::fixed and std::setprecision
      Duration dur = (Clock::now() - start_time_);
      Duration dur2 = (Clock::now() - last_start_time_);

      auto speed = double(progress_) / dur.count();
      auto speed2 = double(progress_ - last_progress_) / dur2.count();

      ss << std::fixed << std::setprecision(2) << "(";
      if (speed_ == Speed::Overall or speed_ == Speed::Both) { ss << speed; }
      if (speed_ == Speed::Both) { ss << " | "; }
      if (speed_ == Speed::Last or speed_ == Speed::Both) { ss << speed2; }
      ss << " " << unit_of_speed_ << ") ";

      out << ss.str();

      last_progress_ = progress_;
      last_start_time_ = Clock::now();
    }
  }

  void start() { start_time_ = Clock::now(); }

  Speedometer(Progress& progress, Speed speed, std::string unit_of_speed)
      : progress_(progress),
        speed_(speed),
        unit_of_speed_(std::move(unit_of_speed)) {}
};

template <typename Progress = size_t>
class CounterDisplay : public AsyncDisplay<CounterDisplay<Progress>> {
 private:
  Progress& progress_; // current amount of work done
  Speedometer<Progress> speedom_;

  void render_counts_(std::ostream& out) {
    std::stringstream ss;
    if (std::is_floating_point<Progress>::value) {
      ss << std::fixed << std::setprecision(2);
    }
    ss << progress_ << " ";
    out << ss.str();
  }

 private:
  void render_(std::ostream& out) {
    this->render_message_(out);
    render_counts_(out);
    speedom_.render_speed(out);
  }

 public:
  CounterDisplay(Progress& progress,
                 std::string message = "",
                 std::string unit_of_speed = "",
                 Speed speed = Speed::None,
                 double period = 0.1,
                 std::ostream& out = std::cout)
      : AsyncDisplay<CounterDisplay<Progress>>(std::move(message), period, out),
        progress_(progress),
        speedom_(progress, speed, std::move(unit_of_speed)) {}

  void start() {
    static_cast<AsyncDisplay<CounterDisplay<Progress>>&>(*this).start();
    speedom_.start();
  }

  friend class AsyncDisplay<CounterDisplay<Progress>>;
  template <typename Left, typename Right>
  friend class Composite;
};

template <typename Progress, typename... Args>
auto Counter(Progress& progress, Args&&... args) {
  return CounterDisplay<Progress>(progress, std::forward<Args>(args)...);
}

template <typename Progress>
class ProgressBarDisplay : public AsyncDisplay<ProgressBarDisplay<Progress>> {
 private:
  Speedometer<Progress> speedom_;
  Progress& progress_;             // work done so far
  const static size_t width_ = 30; // width of progress bar
  size_t total_;                   // total work to be done
  bool counts_;                    // whether to display counts

  const Strings& partials_; // progress bar display strings

  void render_progress_bar_(std::ostream& out) {
    size_t on = width_ * progress_ / total_;
    size_t partial =
        partials_.size() * width_ * progress_ / total_ - partials_.size() * on;
    if (on >= width_) {
      on = width_;
      partial = 0;
    }
    assert(partial != partials_.size());
    size_t off = width_ - on - size_t(partial > 0);

    // draw progress bar
    out << "|";
    for (size_t i = 0; i < on; i++) { out << partials_.back(); }
    if (partial > 0) { out << partials_.at(partial - 1); }
    out << std::string(off, ' ') << "| ";
  }

  void render_counts_(std::ostream& out) {
    if (counts_) { out << progress_ << "/" << total_ << " "; }
  }

  void render_percentage_(std::ostream& out) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss.width(6);
    ss << std::right << progress_ * 100. / total_ << "% ";
    out << ss.str();
  }

  void render_(std::ostream& out) {
    this->render_message_(out);
    render_percentage_(out);
    render_progress_bar_(out);
    render_counts_(out);
    speedom_.render_speed(out);
  }

 public:
  using Style = ProgressBarStyle;

  ProgressBarDisplay(Progress& progress,
                     size_t total,
                     std::string message = "",
                     std::string unit_of_speed = "",
                     Speed speed = Speed::None,
                     bool counts = true,
                     Style style = Blocks,
                     double period = 0.1,
                     std::ostream& out = std::cout)
      : AsyncDisplay<ProgressBarDisplay<Progress>>(std::move(message),
                                                   period,
                                                   out),
        speedom_(progress, speed, std::move(unit_of_speed)),
        progress_(progress),
        total_(total),
        counts_(counts),
        partials_(progress_partials_[static_cast<unsigned short>(style)]) {}

  void start() {
    static_cast<AsyncDisplay<ProgressBarDisplay<Progress>>&>(*this).start();
    speedom_.start();
  }

  friend class AsyncDisplay<ProgressBarDisplay<Progress>>;
  template <typename Left, typename Right>
  friend class Composite;
};

template <typename Progress, typename... Args>
auto ProgressBar(Progress& progress, Args&&... args) {
  return ProgressBarDisplay<Progress>(progress, std::forward<Args>(args)...);
}

} // namespace mew

#endif
