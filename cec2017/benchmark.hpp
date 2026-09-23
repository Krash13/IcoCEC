#pragma once

#include "countries_algorithm_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace bench {

// An objective receives a point and returns its fitness.
// Plain function pointers and capturing lambdas both work here.
typedef std::vector<double> Vec;
typedef std::function<double(const Vec&)> Fitness;

struct Result {
    double success_percent = 0.0;
    double calls = 0.0; // Mean number of evaluations, including initialization.
    double avg_function = 0.0;
    double std_function = 0.0;
    double best_function = std::numeric_limits<double>::infinity();
    Vec best_x;
};

class Benchmark {
public:
    typedef std::function<void(int, int, double, long, bool)> Progress;

    // Create a new algorithm for every independent run.
    Result run(const Fitness& fitness,
               CountriesAlgorithm::Params params,
               double optimum,
               double tolerance,
               int runs,
               long max_calls,
               std::uint64_t seed = 42,
               const Progress& progress = Progress()) const {
        if (!fitness || runs <= 0 || max_calls <= 0 || tolerance < 0.0) {
            throw std::invalid_argument("Benchmark: invalid objective or run settings");
        }

        Result result;
        std::vector<double> values;
        values.reserve(runs);
        long total_calls = 0;
        int successes = 0;

        for (int i = 0; i < runs; ++i) {
            set_random_seed(seed + static_cast<std::uint64_t>(i));
            long calls = 0;
            Fitness counted = [&fitness, &calls](const Vec& x) {
                ++calls;
                return fitness(x);
            };

            // Construction evaluates the initial population; count it too.
            CountriesAlgorithm algorithm(counted, params);
            const std::tuple<Vec, double, long> outcome = algorithm.start(
                Vec(params.x_min.size(), 0.0), 0.0, tolerance, max_calls);
            const Vec& x = std::get<0>(outcome);
            const double value = std::get<1>(outcome);
            const long iterations = std::get<2>(outcome);

            const bool success = std::isfinite(value) &&
                                 std::fabs(value - optimum) <= tolerance;
            if (success) ++successes;
            total_calls += calls;
            values.push_back(value);
            if (value < result.best_function) {
                result.best_function = value;
                result.best_x = x;
            }
            if (progress) progress(i + 1, runs, value, iterations, success);
        }

        result.success_percent = 100.0 * successes / runs;
        result.calls = static_cast<double>(total_calls) / runs;
        for (std::size_t i = 0; i < values.size(); ++i) {
            result.avg_function += values[i] / runs;
        }
        double variance = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double difference = values[i] - result.avg_function;
            variance += difference * difference / runs;
        }
        result.std_function = std::sqrt(variance);
        return result;
    }
};

} // namespace bench
