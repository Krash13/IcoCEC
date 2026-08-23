#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <cstdint>
#include "countries_algorithm.hpp"

namespace bench {

using Vec = std::vector<double>;
using Clock = std::chrono::steady_clock;

inline double l2_norm_diff(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

inline double stddev(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = mean(v);
    double s = 0.0;
    for (double x : v) {
        double d = x - m;
        s += d * d;
    }
    return std::sqrt(s / v.size());
}

template <typename F>
struct CountedFunction {
    F f;
    mutable long calls = 0;

    CountedFunction() = default;
    explicit CountedFunction(F func) : f(std::move(func)) {}

    double operator()(const Vec& x) const {
        ++calls;
        return f(x);
    }

    void reset() const { calls = 0; }
};

struct Result {
    std::string test_function;
    double success_percent = 0.0;
    double calls = 0.0;
    int iterations = 0;
    std::string avg_time;
    double avg_iterations = 0.0;
    double avg_function = 0.0;
    double std_function = 0.0;
    double best_function = 0.0;
    Vec best_x;
    std::map<std::string, Vec> input_args;
};

template <typename Method, typename TestFunc>
class BaseBenchMark {
public:
    using ProgressCallback = std::function<void(int, int, double, long, bool)>;

    BaseBenchMark(std::function<Method()> method_factory,
                  TestFunc& test_function,
                  std::string test_function_name,
                  Vec x_min,
                  Vec x_max,
                  int n,
                  int iterations,
                  double x_eps,
                  std::optional<double> y_eps = std::nullopt,
                  std::optional<long> max_calls = std::nullopt,
                  std::optional<double> target_f = std::nullopt,
                  std::uint64_t base_seed = 42
                )
        : method_factory_(std::move(method_factory)),
          test_function_(test_function),
          test_function_name_(std::move(test_function_name)),
          x_min_(std::move(x_min)), x_max_(std::move(x_max)),
          n_(n), iterations_(iterations),
          x_eps_(x_eps), y_eps_(y_eps), max_calls_(max_calls),
          method_(method_factory_()), target_f_(target_f), base_seed_(base_seed){}

    virtual ~BaseBenchMark() = default;
    virtual Vec canonical_x() const = 0;

    Result operator()(ProgressCallback progress_callback = nullptr) {
        int successes = 0;
        long calls_count = 0;
        long iterations_count = 0;
        std::chrono::duration<double> sum_time{0};
        std::vector<double> f_results;
        f_results.reserve(iterations_);
        double best_f = std::numeric_limits<double>::infinity();
        Vec best_x;

        const Vec cx = canonical_x();

        for (int i = 0; i < iterations_; ++i) {
            set_random_seed(base_seed_ + static_cast<std::uint64_t>(i));

            // Создание ICO после установки seed.
            // В конструкторе CountriesAlgorithm создаётся начальная популяция.
            reset_method();
            auto t0 = Clock::now();
            auto [res_x, res_f, res_it] = method_->start(cx, x_eps_, y_eps_, max_calls_);
            auto t1 = Clock::now();

            iterations_count += res_it;
            sum_time += (t1 - t0);
            double ref_y = target_f_.has_value() ? *target_f_ : test_function_(cx);
            bool success = false;
            if (target_f_.has_value()) {
                // Для CEC2017: успех только по близости к известному оптимуму.
                if (y_eps_.has_value() && std::fabs(res_f - ref_y) <= *y_eps_) {
                    success = true;
                }
            }
            else {
                if (l2_norm_diff(res_x, cx) < x_eps_) {
                    success = true;
                } else if (y_eps_.has_value() && std::fabs(res_f - test_function_(cx)) <= *y_eps_) {
                    success = true;
                } else if (res_f < test_function_(cx)) {
                    success = true;
                }
            }
            if (success) ++successes;

            calls_count += test_function_.calls;
            f_results.push_back(res_f);
            if (res_f < best_f) {
                best_f = res_f;
                best_x = res_x;
            }

            if (progress_callback) progress_callback(i + 1, iterations_, res_f, res_it, success);

            test_function_.reset();
            // reset_method();
        }

        Result result;
        result.test_function = test_function_name_;
        result.success_percent = 100.0 * successes / iterations_;
        result.calls = (double)calls_count / iterations_;
        result.iterations = iterations_;
        result.avg_time = std::to_string(sum_time.count() / iterations_) + " s";
        result.avg_iterations = (double)iterations_count / iterations_;
        result.avg_function = mean(f_results);
        result.std_function = stddev(f_results);
        result.best_function = *std::min_element(f_results.begin(), f_results.end());
        result.best_x = best_x;
        result.input_args["x_min"] = x_min_;
        result.input_args["x_max"] = x_max_;
        return result;
    }

protected:
    void reset_method() { method_.reset(); method_.emplace(method_factory_()); }

    std::function<Method()> method_factory_;
    TestFunc& test_function_;
    std::string test_function_name_;
    Vec x_min_, x_max_;
    int n_;
    std::uint64_t base_seed_;
    int iterations_;
    double x_eps_;
    std::optional<double> y_eps_;
    std::optional<double> target_f_;
    std::optional<long> max_calls_;
    std::optional<Method> method_;
};

#define BENCHMARK_CLASS(CLASS_NAME, TESTFUNC_NAME) \
template <typename Method> \
class CLASS_NAME : public BaseBenchMark<Method, CountedFunction<double(*)(const Vec&)> > { \
public: \
    using TestFunc = CountedFunction<double(*)(const Vec&)>; \
    using Base = BaseBenchMark<Method, TestFunc>; \
    CLASS_NAME(std::function<Method()> method_factory, Vec x_min, Vec x_max, int n, int iterations, double x_eps, std::optional<double> y_eps = std::nullopt, std::optional<long> max_calls = std::nullopt) \
        : Base(std::move(method_factory), test_function_, #TESTFUNC_NAME, std::move(x_min), std::move(x_max), n, iterations, x_eps, y_eps, max_calls), test_function_(TESTFUNC_NAME) {} \
    Vec canonical_x() const override { return Vec(this->x_min_.size(), 0.0); } \
private: \
    TestFunc test_function_; \
};

} // namespace bench
