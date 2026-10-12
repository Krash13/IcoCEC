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

// Start with the algorithm's own defaults, then specify this experiment's
// tuned values in one place. Dimensions, bounds, and target vary per function.
static CountriesAlgorithm::Params experiment_params() {
    CountriesAlgorithm::Params p;
    p.p_min = 6.81035e-05;
    p.p_max = 1.41681;
    p.M = 5;
    p.N = 200;
    p.n_min = 1;
    p.n_max = 2;
    p.m_min = 2;
    p.m_max = 3;
    p.k = 10;
    p.l = 4;
    p.ep_elite = 0.192889;
    p.ep_dead = 0.204716;
    p.max_mutation = 10;
    p.tmax = 2200;
    p.gray_percent = 0.0;
    p.p_war = 0.0;
    p.p_trade = 0.0;
    p.p_motion = 0.0;
    p.p_epidemic = 0.177915;
    p.p_migration = 0.0;
    p.action_alpha = 0.0551293;
    p.action_pmin = 0.05;
    p.action_warmup_frac = 0.12069;
    p.stagnation_limit = 24;
    p.restart_country_frac = 0.146922;
    p.migration_frac = 0.32868;
    p.parent_rank_pressure = 3.0;
    p.real_blx_share_start = 0.0;
    p.real_blx_share_end = 0.0;
    p.real_de_share_start = 1.0;
    p.real_de_share_end = 1.0;
    p.de_f = 0.55;
    p.de_cr = 0.90;
    p.de_pbest_frac = 0.20;
    p.de_pool_frac = 0.60;
    p.de_pool_frac_end = 0.40;
    p.de_exploitation_start_frac = 1.0;
    p.de_exploit_share_start = 1.0;
    p.de_exploit_share_end = 1.0;
    p.de_adaptive = true;
    p.de_memory_size = 5;
    p.de_f_sigma = 0.10;
    p.de_cr_sigma = 0.05;
    p.adaptive_reproduction_operators = true;
    p.operator_alpha = 0.10;
    p.operator_pmin = 0.05;
    p.population_reduction = true;
    p.min_country_size = 20;
    p.min_countries = 1;
    p.gray_uniform_share_start = 0.50;
    p.gray_uniform_share_end = 0.50;
    p.gray_two_point_share_start = 0.50;
    p.gray_two_point_share_end = 0.50;
    p.printing = false;
    return p;
}

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
    CountriesAlgorithm::Params params = experiment_params();
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
