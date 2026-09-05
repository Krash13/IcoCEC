// ============================================================================
// main_cec2017.cpp
// Тестирование CountriesAlgorithm (ICO) на наборе функций CEC2017.
// Размерность задачи задаётся один раз через переменную `dim`
// (поддерживаются значения 2, 10, 20, 30, 50, 100 — как в оригинальном
// C-коде CEC2017), либо передаётся первым аргументом командной строки.
//
// Важно: рядом с исполняемым файлом должна лежать папка input_data/
// (матрицы поворота / векторы сдвига), иначе cec17_test_func не найдёт
// файлы данных и результаты будут некорректны.
// ============================================================================

#include "benchmark.hpp"
#include "countries_algorithm_v2.hpp"
#include "cec2017.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cstdlib>
#include <algorithm>

using namespace bench;
using Vec = std::vector<double>;

struct IcoSettings {
    double p_min        = 0.023;
    double p_max        = 1.8;
    int    M            = 10;
    int    N            = 11;
    int    n_min        = 1;
    int    n_max        = 2;
    int    m_min        = 2;
    int    m_max        = 3;
    int    k            = 8;
    int    l            = 2;
    double ep_elite     = 0.077;
    double ep_dead      = 0.28;
    int    max_mutation = 10;
    int    tmax         = 1200;
    double gray_percent = 0.15;
    double p_war      = 0.22;
    double p_trade    = 0.45;
    double p_motion   = 0.3;
    double p_epidemic = 0.07;
    double p_migration = 0.1;
    double action_alpha         = 0.321974;
    double action_pmin          = 0.05;
    double action_warmup_frac   = 0.00759028;
    int stagnation_limit     = 17;
    double restart_country_frac = 0.236802;
    double migration_frac       = 0.212646;
    double parent_rank_pressure = 3.0;
    double real_blx_share_start   = 0.35;
    double real_blx_share_end     = 0.15;
    double real_eigen_share_start = 0.35;
    double real_eigen_share_end   = 0.15;
    double real_de_share_start    = 0.30;
    double real_de_share_end      = 0.70;
    double de_f = 0.55;
    double de_cr = 0.90;
    double de_pbest_frac = 0.20;
    bool population_reduction = true;
    int min_country_size = 4;
    int min_countries = 2;
    double gray_uniform_share_start   = 0.05;
    double gray_uniform_share_end     = 0.05;
    double gray_two_point_share_start = 0.05;
    double gray_two_point_share_end   = 0.05;
    double gray_eigen_share_start     = 0.90;
    double gray_eigen_share_end       = 0.90;
    double eigen_ps = 0.50;
    bool   eigen_local_search            = true;
    double eigen_local_search_start_frac = 0.50;
    double eigen_local_sigma_start       = 0.02;
    double eigen_local_sigma_end         = 1e-8;
    double eigen_min_axis_scale          = 1e-3;
    int    eigen_local_trials            = 2;
    bool   eigen_local_stats             = true;
    bool   printing     = false;
    std::vector<int> genes;
};

class CountriesAlgorithmMethod {
public:
    using Vec = std::vector<double>;
    using Params = CountriesAlgorithm::Params;

    CountriesAlgorithmMethod(FuncT func, const Vec& x_min, const Vec& x_max, const Params& params)
        : algo_(func, params)
    {
    }

    std::tuple<Vec, double, long> start(const Vec& canonical_x,
                                        double epsilon,
                                        std::optional<double> y_epsilon,
                                        std::optional<long> max_calls) {
        return algo_.start(canonical_x, epsilon, y_epsilon, max_calls);
    }

private:
    CountriesAlgorithm algo_;
};

using Method = CountriesAlgorithmMethod;
using TestFunc = CountedFunction<double(*)(const Vec&)>;

static Method::Params make_params(const IcoSettings& s, const Vec& x_min, const Vec& x_max, int dim) {
    Method::Params p;
    p.x_min        = x_min;
    p.x_max        = x_max;
    p.p_min        = s.p_min;
    p.p_max        = s.p_max;
    p.M            = s.M;
    p.N            = s.N;
    p.n_min        = s.n_min;
    p.n_max        = s.n_max;
    p.m_min        = s.m_min;
    p.m_max        = s.m_max;
    p.k            = s.k;
    p.l            = s.l;
    p.ep_elite     = s.ep_elite;
    p.ep_dead      = s.ep_dead;
    p.max_mutation = s.max_mutation;
    p.tmax         = s.tmax;
    p.gray_percent = s.gray_percent;
    p.printing     = s.printing;
    p.p_war        = s.p_war;
    p.p_trade      = s.p_trade;
    p.p_motion     = s.p_motion;
    p.p_epidemic   = s.p_epidemic;
    p.p_migration = s.p_migration;
    p.action_alpha = s.action_alpha;
    p.action_pmin = s.action_pmin;
    p.action_warmup_frac = s.action_warmup_frac;
    p.stagnation_limit = s.stagnation_limit;
    p.restart_country_frac = s.restart_country_frac;
    p.migration_frac = s.migration_frac;
    p.parent_rank_pressure = s.parent_rank_pressure;
    p.real_blx_share_start = s.real_blx_share_start;
    p.real_blx_share_end = s.real_blx_share_end;
    p.real_eigen_share_start = s.real_eigen_share_start;
    p.real_eigen_share_end = s.real_eigen_share_end;
    p.real_de_share_start = s.real_de_share_start;
    p.real_de_share_end = s.real_de_share_end;
    p.de_f = s.de_f;
    p.de_cr = s.de_cr;
    p.de_pbest_frac = s.de_pbest_frac;
    p.population_reduction = s.population_reduction;
    p.min_country_size = s.min_country_size;
    p.min_countries = s.min_countries;
    p.gray_uniform_share_start = s.gray_uniform_share_start;
    p.gray_uniform_share_end = s.gray_uniform_share_end;
    p.gray_two_point_share_start = s.gray_two_point_share_start;
    p.gray_two_point_share_end = s.gray_two_point_share_end;
    p.gray_eigen_share_start = s.gray_eigen_share_start;
    p.gray_eigen_share_end = s.gray_eigen_share_end;
    p.eigen_ps = s.eigen_ps;
    p.eigen_local_search = s.eigen_local_search;
    p.eigen_local_search_start_frac = s.eigen_local_search_start_frac;
    p.eigen_local_sigma_start = s.eigen_local_sigma_start;
    p.eigen_local_sigma_end = s.eigen_local_sigma_end;
    p.eigen_min_axis_scale = s.eigen_min_axis_scale;
    p.eigen_local_trials = s.eigen_local_trials;
    p.eigen_local_stats = s.eigen_local_stats;
    p.genes        = std::vector<int>(dim, 32);
    return p;
}

// Запуск ICO на одной функции CEC2017.
// funcid  — номер функции (1..30).
// name    — человекочитаемое имя (для вывода).
// func    — указатель на cec2017::fN.
// dim     — размерность задачи (2, 10, 20, 30, 50 или 100).
Result run_one_cec(int funcid,
                    const std::string& name,
                    cec2017::FuncPtr func,
                    int dim,
                    const IcoSettings& s,
                    int iterations,
                    std::uint64_t base_seed,
                    std::optional<long> max_calls) {
    Vec x_min(dim, cec2017::kLowerBound);
    Vec x_max(dim, cec2017::kUpperBound);
    // Истинный оптимум x* нам не известен (он скрыт в shift-данных),
    // поэтому canonical_x используется только для расчёта success
    // через сравнение значений функции (y_eps), не через L2-расстояние.
    Vec canonical_x(dim, 0.0); //только для стандартных non-shifted функций

    auto counted = TestFunc(func);
    auto params = make_params(s, x_min, x_max, dim);


    auto method_factory = [=, &counted]() mutable {
        return Method(FuncT(std::ref(counted)), x_min, x_max, params);
    };

    class LocalBench : public BaseBenchMark<Method, TestFunc> {
        Vec can_x_;
    public:
        LocalBench(std::function<Method()> f,
                   TestFunc& tf,
                   std::string name,
                   Vec xmin,
                   Vec xmax,
                   Vec can_x,
                   int n,
                   int iterations,
                   double x_eps,
                   std::optional<double> y_eps,
                   std::optional<long> max_calls,
                   std::optional<double> target_f = std::nullopt,
                    std::uint64_t base_seed = 42)
            : BaseBenchMark<Method, TestFunc>(std::move(f), tf, std::move(name), std::move(xmin), std::move(xmax), n, iterations, x_eps, y_eps, max_calls, target_f, base_seed),
              can_x_(std::move(can_x)) {}
        Vec canonical_x() const override { return can_x_; }
    };

    // x_eps произвольный и не используем его, т.к. истинный минимум
    // неизвестен в координатах x; критерий успеха — по значению функции (y_eps).
    double x_eps = 1e-9;
    double y_eps = 1e-8; // допуск по значению функции относительно оптимума funcid*100
    double true_optimum = cec2017::optimum_value(funcid); // funcid * 100

    LocalBench bm(method_factory, counted, name, x_min, x_max, canonical_x,
                  dim, iterations, x_eps, y_eps, max_calls, true_optimum, base_seed);

    auto on_progress = [&](int idx, int total, double res_f, long res_it, bool success) {
        double err = res_f - cec2017::optimum_value(funcid);
        std::cout << "  [" << name << "] прогон " << idx << "/" << total
                  << ": f=" << res_f
                  << ", error=" << err
                  << ", итераций=" << res_it
                  << ", успех=" << (success ? "да" : "нет")
                  << std::endl;
    };

    Result r = bm(on_progress);
    double err_best = r.best_function - cec2017::optimum_value(funcid);
    double err_avg  = r.avg_function - cec2017::optimum_value(funcid);
    std::cout << name << ": {"
              << "dim=" << dim
              << ", success_percent=" << r.success_percent
              << ", calls=" << r.calls
              << ", avg_error=" << err_avg
              << ", std_function=" << r.std_function
              << ", best_error=" << err_best
              << "}" << std::endl << std::endl;
    return r;
}

int main(int argc, char** argv) {
    // --- Задайте размерность здесь, либо передайте первым аргументом CLI ---
    int dim = 2; // допустимые значения: 2, 10, 20, 30, 50, 100
    if (argc > 1) dim = std::atoi(argv[1]);

    static const int allowed[] = {2, 10, 20, 30, 50, 100};
    if (std::find(std::begin(allowed), std::end(allowed), dim) == std::end(allowed)) {
        std::cerr << "Ошибка: размерность dim=" << dim
                  << " не поддерживается. Допустимые значения: 2, 10, 20, 30, 50, 100."
                  << std::endl;
        return 1;
    }

    const int iterations = 20; // число независимых прогонов на каждую функцию

    // Официальный протокол CEC2017: max_evals = 10000 * dim.
    std::optional<long> max_calls = 10000L * dim;

    IcoSettings s;


s.p_min        = 6.81035e-05;
s.p_max        = 1.41681;
s.M            = 10;
s.N            = 10;
s.n_min        = 1;
s.n_max        = 2;
s.m_min        = 2;
s.m_max        = 3;
s.k            = 10;
s.l            = 4;
s.ep_elite     = 0.192889;
s.ep_dead      = 0.204716;
s.max_mutation = 10;
s.gray_percent = 0.35;
/*
s.gray_percent = 0.773545;
//s.real_blx_share_start   = 0.868416;
//s.real_blx_share_end     = 0.941535;
//s.real_eigen_share_start = 0.131584;
//s.real_eigen_share_end   = 0.0584651;
s.real_blx_share_start   = 0.35;
s.real_blx_share_end     = 0.15;
    s.real_eigen_share_start = 0.35;
    s.real_eigen_share_end   = 0.15;
    s.real_de_share_start    = 0.30;
    s.real_de_share_end      = 0.70;

s.gray_uniform_share_start   = 0.461211;
s.gray_uniform_share_end     = 0.0796675;
s.gray_two_point_share_start = 0.0921363;
s.gray_two_point_share_end   = 0.689922;
s.gray_eigen_share_start     = 0.446653;
s.gray_eigen_share_end       = 0.230411;
s.eigen_ps                   = 0.0750305;
*/
s.p_motion     = 0.317617;
s.p_trade      = 0.18145;
s.p_war        = 0.174599;
s.p_epidemic   = 0.177915;
s.p_migration  = 0.14842;

s.action_alpha         = 0.0551293;
s.action_pmin          = 0.05;
s.action_warmup_frac   = 0.12069;
s.stagnation_limit     = 24;
s.restart_country_frac = 0.146922;
s.migration_frac       = 0.32868;

    s.genes = std::vector<int>(dim, 32);

    std::cout << "=== CEC2017 benchmark, dim=" << dim
              << ", max_calls=" << *max_calls
              << ", iterations=" << iterations << " ===" << std::endl;
    for (const auto& entry : cec2017::all_functions()) {
        // F2 исторически исключена из официального сравнения CEC2017
        // (несогласованность реализаций между платформами) — раскомментируйте,
        // если хотите исключить ее
        // if (entry.id == 2) continue;

        // Для D=2 нет матриц поворота для функций hf01-hf10 и cf09-cf10.
        // Оригинальный код выведет предупреждение в консоль
        // и продолжит работу с некорректными данными, поэтому пропускаем их.
        if (dim == 2 && ((entry.id >= 11 && entry.id <= 20) || entry.id == 29 || entry.id == 30)) {
            continue;
        }

        run_one_cec(entry.id, entry.name, entry.func, dim, s, iterations, 42, max_calls);
    }

    return 0;
}
