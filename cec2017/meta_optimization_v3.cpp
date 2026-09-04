// ============================================================================
// meta_optimization_v4.cpp
//
// Мета-оптимизация параметров ICO (CountriesAlgorithm) с помощью GWO.
//
// Обновления и изменения:
//   1) Поддержка 5 действий: Motion, Trade, War, Epidemic, Migration.
//      Логиты действий нормализуются через Softmax (сумма строго 1.0).
//   2) Добавлены параметры периодического глобального рестарта и миграции:
//      stagnation_limit, restart_country_frac, migration_frac.
//   3) Границы поиска Xmax строятся относительно Xmin через оператор + (delta),
//      чтобы наглядно видеть интервал варьирования каждого параметра.
//   4) Оценка на репрезентативном наборе CEC2017 (F1, F6, F13, F21, F29)
//      с логарифмической нормировкой log1p(error) и усреднением по REPEATS.
//   5) Итоговый вывод параметров в формате структуры IcoSettings.
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
constexpr int           META_DIM       = 24;    // 24 параметра ICO
constexpr int           ST_DIM         = 10;    // размерность тестовой задачи CEC2017
constexpr long          ICO_MAX_CALLS  = 20000; // бюджет вызовов ЦФ на один прогон ICO
constexpr int           REPEATS        = 3;     // прогонов ICO на каждый набор параметров
constexpr int           GWO_POP        = 50;
constexpr int           GWO_ITERS      = 100;

// Репрезентативный набор функций CEC2017
struct BenchFunc {
    int id;
    cec2017::FuncPtr func;
};

static const std::vector<BenchFunc> kBenchSet = {
    {1,  cec2017::f1},   // Bent Cigar (унимодальная, плохо обусловленная)
    {6,  cec2017::f6},   // Expanded Scaffer F6 (мультимодальная)
    {13, cec2017::f13},  // Hybrid Function 3
    {21, cec2017::f21},  // Composition Function 1
    {29, cec2017::f29},  // Composition Function 9
};

// ----------------------------------------------------------------------------
// Декодирование вектора GWO (x, размер META_DIM=24) в Params ICO
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

    // Доли операторов кроссовера меняются линейно по FEs / MaxFEs.
    // Пока они не включены в META_DIM и задаются здесь явно.
    p.real_blx_share_start   = 0.50;
    p.real_blx_share_end     = 0.25;
    p.real_eigen_share_start = 0.50;
    p.real_eigen_share_end   = 0.75;
    p.gray_uniform_share_start   = 0.05;
    p.gray_uniform_share_end     = 0.05;
    p.gray_two_point_share_start = 0.05;
    p.gray_two_point_share_end   = 0.05;
    p.gray_eigen_share_start     = 0.90;
    p.gray_eigen_share_end       = 0.90;
    p.eigen_ps = 0.50;

    // Softmax по 5 логитам действий: motion, trade, war, epidemic, migration
    std::array<double, 5> logits = {x[14], x[15], x[16], x[17], x[18]};
    double max_logit = *std::max_element(logits.begin(), logits.end());
    std::array<double, 5> exps;
    double sum_exp = 0.0;
    for (int i = 0; i < 5; ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum_exp += exps[i];
    }

    p.p_motion    = exps[0] / sum_exp;
    p.p_trade     = exps[1] / sum_exp;
    p.p_war       = exps[2] / sum_exp;
    p.p_epidemic  = exps[3] / sum_exp;
    p.p_migration = exps[4] / sum_exp;

    // Нормализация суммы
    double sum_p = p.p_motion + p.p_trade + p.p_war + p.p_epidemic + p.p_migration;
    p.p_motion    /= sum_p;
    p.p_trade     /= sum_p;
    p.p_war       /= sum_p;
    p.p_epidemic  /= sum_p;
    p.p_migration /= sum_p;

    p.adaptive_actions   = true;
    p.action_alpha       = std::clamp(x[19], 0.01, 0.9);
    p.action_pmin        = 0.05;
    p.action_warmup_frac = std::clamp(x[20], 0.0, 0.5);

    // Новые параметры рестарта и миграции
    p.stagnation_limit     = std::max(5, static_cast<int>(std::round(x[21])));
    p.restart_country_frac = std::clamp(x[22], 0.05, 0.50);
    p.migration_frac       = std::clamp(x[23], 0.05, 0.60);

    p.x_min = Vec(dim, cec2017::kLowerBound);
    p.x_max = Vec(dim, cec2017::kUpperBound);
    p.genes = std::vector<int>(dim, 32);
    p.tmax  = 300;
    p.printing = false;

    return p;
}

// ----------------------------------------------------------------------------
// Fitness-функция для GWO
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
            return -std::numeric_limits<double>::infinity();
        }

        double total_error = 0.0;
        int n_terms = 0;

        for (const auto& bf : kBenchSet) {
            double optimum = cec2017::optimum_value(bf.id);
            double sum_err_this_func = 0.0;

            for (int r = 0; r < REPEATS; ++r) {
                set_random_seed(1000u * static_cast<unsigned>(bf.id) + r + 1);

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
                    sum_err_this_func += 1e12; // штраф
                }
            }

            double avg_err_this_func = sum_err_this_func / REPEATS;
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

    // 1. Задаем нижние границы поиска Xmin (24 параметра)
    Vec Xmin = {
        0.000001, // 0:  p_min_raw
        1.0,      // 1:  p_max_raw
        10.0,     // 2:  M (число стран)
        10.0,     // 3:  N (размер страны)
        1.0,      // 4:  n_min_raw
        2.0,      // 5:  n_max_raw
        1.0,      // 6:  m_min_raw
        1.0,      // 7:  m_max_raw
        2.0,      // 8:  k (Trade)
        2.0,      // 9:  l (War)
        0.05,     // 10: ep_elite
        0.15,     // 11: ep_dead
        10.0,     // 12: max_mutation
        0.0,      // 13: gray_percent
        -2.0,     // 14: ложит p_motion
        -2.0,     // 15: ложит p_trade
        -2.0,     // 16: ложит p_war
        -2.0,     // 17: ложит p_epidemic
        -2.0,     // 18: ложит p_migration
        0.02,     // 19: action_alpha
        0.0,      // 20: action_warmup_frac
        10.0,     // 21: stagnation_limit
        0.05,     // 22: restart_country_frac
        0.10      // 23: migration_frac
    };

    // 2. Дельты (размах диапазона поиска) для каждого параметра
    Vec Delta = {
        0.499999, // 0:  p_min -> Xmin[0] + 0.499999 = 0.5
        1.0,      // 1:  p_max -> Xmin[1] + 1.0 = 2.0
        30.0,     // 2:  M     -> Xmin[2] + 30.0 = 40.0
        30.0,     // 3:  N     -> Xmin[3] + 30.0 = 40.0
        9.0,      // 4:  n_min -> Xmin[4] + 9.0 = 10.0
        18.0,     // 5:  n_max -> Xmin[5] + 18.0 = 20.0
        9.0,      // 6:  m_min -> Xmin[6] + 9.0 = 10.0
        14.0,     // 7:  m_max -> Xmin[7] + 14.0 = 15.0
        10.0,     // 8:  k     -> Xmin[8] + 10.0 = 12.0
        6.0,      // 9:  l     -> Xmin[9] + 6.0 = 8.0
        0.30,     // 10: ep_elite -> Xmin[10] + 0.30 = 0.35
        0.35,     // 11: ep_dead  -> Xmin[11] + 0.35 = 0.50
        30.0,     // 12: max_mutation -> Xmin[12] + 30.0 = 40.0
        1.0,      // 13: gray_percent -> Xmin[13] + 1.0 = 1.0
        4.0,      // 14: ложит p_motion -> Xmin[14] + 4.0 = +2.0
        4.0,      // 15: ложит p_trade  -> Xmin[15] + 4.0 = +2.0
        4.0,      // 16: ложит p_war    -> Xmin[16] + 4.0 = +2.0
        4.0,      // 17: ложит p_epidemic  -> Xmin[17] + 4.0 = +2.0
        4.0,      // 18: ложит p_migration -> Xmin[18] + 4.0 = +2.0
        0.58,     // 19: action_alpha -> Xmin[19] + 0.58 = 0.60
        0.30,     // 20: action_warmup_frac -> Xmin[20] + 0.30 = 0.30
        40.0,     // 21: stagnation_limit -> Xmin[21] + 40.0 = 50.0
        0.35,     // 22: restart_country_frac -> Xmin[22] + 0.35 = 0.40
        0.40      // 23: migration_frac -> Xmin[23] + 0.40 = 0.50
    };

    // 3. Вычисление верхних границ Xmax относительно Xmin через оператор +
    Vec Xmax(META_DIM);
    for (size_t i = 0; i < META_DIM; ++i) {
        Xmax[i] = Xmin[i] + Delta[i];
    }

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
    std::cout << "s.real_blx_share_start   = " << final_params.real_blx_share_start   << ";\n";
    std::cout << "s.real_blx_share_end     = " << final_params.real_blx_share_end     << ";\n";
    std::cout << "s.real_eigen_share_start = " << final_params.real_eigen_share_start << ";\n";
    std::cout << "s.real_eigen_share_end   = " << final_params.real_eigen_share_end   << ";\n";
    std::cout << "s.gray_uniform_share_start   = " << final_params.gray_uniform_share_start   << ";\n";
    std::cout << "s.gray_uniform_share_end     = " << final_params.gray_uniform_share_end     << ";\n";
    std::cout << "s.gray_two_point_share_start = " << final_params.gray_two_point_share_start << ";\n";
    std::cout << "s.gray_two_point_share_end   = " << final_params.gray_two_point_share_end   << ";\n";
    std::cout << "s.gray_eigen_share_start     = " << final_params.gray_eigen_share_start     << ";\n";
    std::cout << "s.gray_eigen_share_end       = " << final_params.gray_eigen_share_end       << ";\n";
    std::cout << "s.eigen_ps                   = " << final_params.eigen_ps                   << ";\n";
    std::cout << "s.p_motion     = " << final_params.p_motion       << ";\n";
    std::cout << "s.p_trade      = " << final_params.p_trade        << ";\n";
    std::cout << "s.p_war        = " << final_params.p_war          << ";\n";
    std::cout << "s.p_epidemic   = " << final_params.p_epidemic     << ";\n";
    std::cout << "s.p_migration  = " << final_params.p_migration    << ";\n";
    std::cout << "s.adaptive_actions     = " << (final_params.adaptive_actions ? "true" : "false") << ";\n";
    std::cout << "s.action_alpha         = " << final_params.action_alpha         << ";\n";
    std::cout << "s.action_pmin          = " << final_params.action_pmin          << ";\n";
    std::cout << "s.action_warmup_frac   = " << final_params.action_warmup_frac   << ";\n";
    std::cout << "s.stagnation_limit     = " << final_params.stagnation_limit     << ";\n";
    std::cout << "s.restart_country_frac = " << final_params.restart_country_frac << ";\n";
    std::cout << "s.migration_frac       = " << final_params.migration_frac       << ";\n";

    return 0;
}
