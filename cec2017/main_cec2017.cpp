// Benchmark ICO against the selected CEC2017 functions.
// Run from the directory containing input_data/: ./ico_cec2017 [dimension]
#include "benchmark.hpp"
#include "cec2017.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

typedef std::vector<double> Vec;

int main(int argc, char** argv) {
    int dim = 2;
    if (argc > 1) dim = std::atoi(argv[1]);
    const int allowed[] = {2, 10, 20, 30, 50, 100};
    if (std::find(std::begin(allowed), std::end(allowed), dim) ==
        std::end(allowed)) {
        std::cerr << "Unsupported dimension: " << dim << std::endl;
        return 1;
    }

    const int runs = 20;
    const long max_calls = 10000L * dim;
    const double tolerance = 1e-8; // Absolute error relative to known F(x*).
    CountriesAlgorithm::Params params;
    params.printing = false;
    params.x_min.assign(dim, cec2017::kLowerBound);
    params.x_max.assign(dim, cec2017::kUpperBound);
    params.genes.assign(dim, 32);

    bench::Benchmark benchmark;
    double total_success = 0.0;
    int tested = 0;
    std::cout << "CEC2017: dim=" << dim << ", max_calls=" << max_calls
              << ", runs=" << runs << std::endl;

    for (const cec2017::FunctionEntry& entry : cec2017::all_functions()) {
        // F2 is excluded from official comparisons. Uncomment to skip it.
        // if (entry.id == 2) continue;
        // These functions lack the required input data for D=2.
        if (dim == 2 && ((entry.id >= 11 && entry.id <= 20) ||
                         entry.id == 29 || entry.id == 30)) continue;

        // Keep the original experiment's short subset; remove this condition
        // to benchmark every applicable function.
        if (entry.id != 1 && entry.id != 3 &&
            entry.id != 6 && entry.id != 11) continue;

        const double optimum = cec2017::optimum_value(entry.id);
        params.target_f = optimum;
        bench::Result result = benchmark.run(
            entry.func, params, optimum, tolerance, runs, max_calls, 42,
            [&entry, optimum](int index, int total, double value,
                              long iterations, bool success) {
                std::cout << "  [" << entry.name << "] run " << index
                          << "/" << total << ": f=" << value
                          << ", error=" << value - optimum
                          << ", iterations=" << iterations
                          << ", success=" << (success ? "yes" : "no")
                          << std::endl;
            });
        std::cout << entry.name << ": dim=" << dim
                  << ", success_percent=" << result.success_percent
                  << ", calls=" << result.calls
                  << ", avg_error=" << result.avg_function - optimum
                  << ", std_function=" << result.std_function
                  << ", best_error=" << result.best_function - optimum
                  << std::endl << std::endl;
        total_success += result.success_percent;
        ++tested;
    }

    std::cout << "Mean success rate: "
              << (tested ? total_success / tested : 0.0) << std::endl;
    return 0;
}
