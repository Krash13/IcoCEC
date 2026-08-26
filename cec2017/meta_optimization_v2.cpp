// ============================================================================
// meta_optimization_v2.cpp
//
// Мета-оптимизация параметров ICO (CountriesAlgorithm) с помощью GWO.
//
// Отличия от исходного meta_optimization.cpp:
//   1) Настройка ведётся не по одной функции (F1), а по РЕПРЕЗЕНТАТИВНОМУ
//      набору функций CEC2017 (унимодальная / простая мультимодальная /
//      гибридная / композитная), fitness = средний нормированный error
//      по набору. Это снижает риск переобучения параметров под одну
//      конкретную форму ландшафта.
//   2) В вектор оптимизируемых параметров добавлены p_war/p_trade/p_motion/
//      p_epidemic (через softmax-параметризацию, чтобы сумма всегда 1.0)
//      и параметры адаптации action_alpha/action_pmin/action_warmup_frac
//      (см. ActionAdaptation в countries_algorithm.hpp).
//   3) Для устойчивости оценки каждый набор параметров ICO прогоняется
//      несколько раз (REPEATS) с разными seed, fitness = среднее по прогонам.
//   4) Итоговые параметры печатаются в виде готовой структуры IcoSettings,
//      которую можно вставить напрямую в main_cec2017.cpp.
//
// Компилировать вместе с остальными файлами проекта (countries_algorithm.hpp,
// cec2017.hpp/.c, evolutionary_algorithms.hpp) через тот же CMakeLists.txt.
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <optional>
#include <memory>
#include <numeric>
#include <limits>
#include <array>
#include <string>

#include "cec2017.hpp"
#include "countries_algorithm_v2.hpp"
#include "evolutionary_algorithms.hpp"

using Vec = std::vector<double>;

// ----------------------------------------------------------------------------
// Конфигурация мета-оптимизации
// ----------------------------------------------------------------------------

constexpr std::uint32_t GWO_SEED       = 123456789u;
constexpr int           META_DIM       = 20;   // размер вектора параметров ICO (см. ниже)
constexpr int           ST_DIM         = 10;   // размерность тестовой задачи (2,10,20,30,50,100)
constexpr long          ICO_MAX_CALLS  = 20000; // бюджет вызовов ЦФ на один прогон ICO
constexpr int           REPEATS        = 3;     // прогонов ICO на каждый набор параметров (усреднение шума)
constexpr int           GWO_POP        = 50;
constexpr int           GWO_ITERS      = 50;

// Репрезентативные функции CEC2017:
//   F1  - Bent Cigar (унимодальная)
//   F6  - Shifted and Rotated Rosenbrock (простая мультимодальная)
//   F13 - Hybrid Function 3
//   F21 - Composition Function 1
//   F29 - Composition Function 9 (сильно мультимодальная)
struct BenchFunc {
    int id;
    cec2017::FuncPtr func;
};

static const std::vector<BenchFunc> kBenchSet = {
    {1,  cec2017::f1},
    {6,  cec2017::f6},
    {13, cec2017::f13},
    {21, cec2017::f21},
    {29, cec2017::f29},
};

// ----------------------------------------------------------------------------
// Декодирование вектора GWO (x, размер META_DIM) в Params ICO.
//
// Индексы:
//   0: p_min_raw, 1: p_max_raw           -> p_min = min, p_max = max
//   2: M                                  -> число стран
//   3: N                                  -> размер страны
//   4: n_min_raw, 5: n_max_raw            -> репродукция
//   6: m_min_raw, 7: m_max_raw            -> вымирание
//   8: k                                  -> размер обмена (Trade)
//   9: l                                  -> размер войны (War)
//   10: ep_elite
//   11: ep_dead
//   12: max_mutation
//   13: gray_percent
//   14-17: логиты действий (war, trade, motion, epidemic) -> softmax -> p_*
//   18: action_alpha       (EMA скорость обучения адаптации действий)
//   19: action_warmup_frac (доля tmax на разогрев адаптации действий)
// ----------------------------------------------------------------------------
static CountriesAlgorithm::Params decode_params(const Vec& x, int dim) {
    CountriesAlgorithm::Params p;

    p.p_min = std::min(x[0], x[1]);
    p.p_max = std::max(x[0], x[1]);
    p.M     = std::max(2, static_cast<int>(std::round(x[2])));
    p.N     = std::max(2, static_cast<int>(std::round(x[3])));
    p.n_min = std::max(1, static_cast<int>(std::round(std::min(x[4], x[5]))));
    p.n_max = std::max(1, static_cast<int>(std::round(std::max(x[4], x[5]))));
    p.m_min = std::max(1, static_cast<int>(std::round(std::min(x[6], x[7]))));
    p.m_max = std::max(1, static_cast<int>(std::round(std::max(x[6], x[7]))));
    p.k     = std::max(1, static_cast<int>(std::round(x[8])));
    p.l     = std::max(1, static_cast<int>(std::round(x[9])));

    p.ep_elite     = std::clamp(x[10], 0.01, 0.9);
    p.ep_dead      = std::clamp(x[11], 0.01, 0.9);
    p.max_mutation = std::max(1, static_cast<int>(std::round(x[12])));
    p.gray_percent = std::clamp(x[13], 0.0, 1.0);

    // Softmax по 4 логитам -> вероятности действий, сумма гарантированно 1.0
    std::array<double, 4> logits = {x[14], x[15], x[16], x[17]};
    double max_logit = *std::max_element(logits.begin(), logits.end());
    std::array<double, 4> exps;
    double sum_exp = 0.0;
    for (int i = 0; i < 4; ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum_exp += exps[i];
    }
    p.p_war      = exps[0] / sum_exp;
    p.p_trade    = exps[1] / sum_exp;
    p.p_motion   = exps[2] / sum_exp;
    p.p_epidemic = exps[3] / sum_exp;

    // Нормировка суммы на случай погрешности округления (Params требует ~1.0)
    double sum_p = p.p_war + p.p_trade + p.p_motion + p.p_epidemic;
    p.p_war      /= sum_p;
    p.p_trade    /= sum_p;
    p.p_motion   /= sum_p;
    p.p_epidemic /= sum_p;

    p.adaptive_actions   = true;
    p.action_alpha       = std::clamp(x[18], 0.01, 0.9);
    p.action_pmin        = 0.05;
    p.action_warmup_frac = std::clamp(x[19], 0.0, 0.5);

    p.x_min = Vec(dim, cec2017::kLowerBound);
    p.x_max = Vec(dim, cec2017::kUpperBound);
    p.genes = std::vector<int>(dim, 32);
    p.tmax  = 300;
    p.printing = false;

    return p;
}

// ----------------------------------------------------------------------------
// Fitness-функция для GWO: усреднённый нормированный error по набору
// функций CEC2017 (kBenchSet), для устойчивости усредняется по REPEATS
// независимым прогонам ICO с разными сидами.
// ----------------------------------------------------------------------------
struct MetaFitness {
    mutable long eval_calls = 0;
    mutable long calls = 0;
    double operator()(const Vec& x) const {
        ++eval_calls;

        const bool sentinel =
            x.size() == META_DIM &&
            std::all_of(x.begin(), x.end(), [](double v) { return std::isnan(v); });
        if (sentinel) {
            // Служебный canonical_x для GWO — заведомо "невозможно хороший" fitness,
            // чтобы условие досрочной остановки внутри BaseAlgorithm никогда не сработало.
            return -std::numeric_limits<double>::infinity();
        }

        double total_error = 0.0;
        int n_terms = 0;

        for (const auto& bf : kBenchSet) {
            double optimum = cec2017::optimum_value(bf.id);

            double sum_err_this_func = 0.0;
            for (int r = 0; r < REPEATS; ++r) {
                rng_engine.seed(1000u * static_cast<unsigned>(bf.id) + r + 1);

                auto params = decode_params(x, ST_DIM);

                auto wrapped = [&bf](const Vec& point) -> double {
                    const bool ico_sentinel =
                        point.size() == static_cast<size_t>(ST_DIM) &&
                        std::all_of(point.begin(), point.end(),
                                    [](double v) { return std::isnan(v); });
                    if (ico_sentinel)
                        return -std::numeric_limits<double>::infinity();
                    return bf.func(point);
                };

                try {
                    CountriesAlgorithm ico(wrapped, params);
                    Vec ico_canonical(ST_DIM, std::numeric_limits<double>::quiet_NaN());
                    auto [best_x, best_f, iters] =
                        ico.start(ico_canonical, 0.0, std::nullopt, ICO_MAX_CALLS);
                    (void)best_x; (void)iters;

                    double err = std::max(0.0, best_f - optimum);
                    sum_err_this_func += err;
                } catch (const std::exception&) {
                    sum_err_this_func += 1e12; // штраф за невалидную конфигурацию
                }
            }

            double avg_err_this_func = sum_err_this_func / REPEATS;

            // Логарифмическая нормировка: ошибки разных функций CEC2017
            // отличаются на порядки, без нормировки одна "тяжёлая" функция
            // задавит вклад остальных в суммарный fitness.
            total_error += std::log1p(avg_err_this_func);
            ++n_terms;
        }

        return total_error / std::max(1, n_terms);
    }
};

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main() {
    std::cout << "=== Мета-оптимизация параметров ICO (GWO) по набору CEC2017 ===\n";
    std::cout << "Размерность задачи: " << ST_DIM
              << "; бюджет ICO на прогон: " << ICO_MAX_CALLS
              << "; повторов на функцию: " << REPEATS << "\n";
    std::cout << "Набор функций: ";
    for (const auto& bf : kBenchSet) std::cout << "F" << bf.id << " ";
    std::cout << "\n\n";

    // Границы поиска для 20-мерного вектора параметров ICO.
    Vec Xmin = {
        0.000001, 1.0,     // p_min_raw, p_max_raw
        10.0, 10.0,        // M, N
        1.0, 2.0,          // n_min_raw, n_max_raw
        1.0, 1.0,          // m_min_raw, m_max_raw
        2.0, 2.0,          // k, l
        0.05, 0.15,        // ep_elite, ep_dead
        10.0,              // max_mutation
        0.0,               // gray_percent
        -2.0, -2.0, -2.0, -2.0, // логиты p_war, p_trade, p_motion, p_epidemic
        0.05,              // action_alpha
        0.0                // action_warmup_frac
    };

    Vec Xmax = {
        0.5, 2.0,
        40.0, 40.0,
        10.0, 20.0,
        10.0, 15.0,
        12.0, 8.0,
        0.35, 0.5,
        40.0,
        1.0,
        2.0, 2.0, 2.0, 2.0,
        0.6,
        0.3
    };

    MetaFitness fitness_function;

    evo::rng::engine().seed(GWO_SEED);

    evo::GreyWolfOptimizer<MetaFitness> gwo(
        fitness_function,
        Xmin,
        Xmax,
        GWO_POP,
        GWO_ITERS,
        /*printing=*/true
    );

    Vec gwo_canonical(META_DIM, std::numeric_limits<double>::quiet_NaN());
    auto [best_params, best_fitness, iterations] =
        gwo.start(gwo_canonical, 0.0, std::nullopt, std::nullopt);

    std::cout << "\n=== Мета-оптимизация завершена ===\n";
    std::cout << "Итераций GWO: " << iterations << "\n";
    std::cout << "Лучший (усреднённый лог-нормированный) fitness: " << best_fitness << "\n\n";

    auto final_params = decode_params(best_params, ST_DIM);

    std::cout << "Найденные параметры ICO (готовы для вставки в IcoSettings):\n\n";
    std::cout << "IcoSettings s;\n";
    std::cout << "s.p_min        = " << final_params.p_min        << ";\n";
    std::cout << "s.p_max        = " << final_params.p_max        << ";\n";
    std::cout << "s.M            = " << final_params.M            << ";\n";
    std::cout << "s.N            = " << final_params.N            << ";\n";
    std::cout << "s.n_min        = " << final_params.n_min        << ";\n";
    std::cout << "s.n_max        = " << final_params.n_max        << ";\n";
    std::cout << "s.m_min        = " << final_params.m_min        << ";\n";
    std::cout << "s.m_max        = " << final_params.m_max        << ";\n";
    std::cout << "s.k            = " << final_params.k            << ";\n";
    std::cout << "s.l            = " << final_params.l            << ";\n";
    std::cout << "s.ep_elite     = " << final_params.ep_elite     << ";\n";
    std::cout << "s.ep_dead      = " << final_params.ep_dead      << ";\n";
    std::cout << "s.max_mutation = " << final_params.max_mutation << ";\n";
    std::cout << "s.gray_percent = " << final_params.gray_percent << ";\n";
    std::cout << "s.p_war        = " << final_params.p_war        << ";\n";
    std::cout << "s.p_trade      = " << final_params.p_trade      << ";\n";
    std::cout << "s.p_motion     = " << final_params.p_motion     << ";\n";
    std::cout << "s.p_epidemic   = " << final_params.p_epidemic   << ";\n";
    std::cout << "s.adaptive_actions   = " << final_params.adaptive_actions   << ";\n";
    std::cout << "s.action_alpha       = " << final_params.action_alpha       << ";\n";
    std::cout << "s.action_pmin        = " << final_params.action_pmin        << ";\n";
    std::cout << "s.action_warmup_frac = " << final_params.action_warmup_frac << ";\n";

    return 0;
}