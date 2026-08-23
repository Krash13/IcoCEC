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

#include "cec2017.hpp"
#include "countries_algorithm.hpp"
#include "evolutionary_algorithms.hpp"

using Vec = std::vector<double>;

constexpr std::uint32_t GWO_SEED = 123456789u;
constexpr std::uint64_t ICO_SEED = 42;
constexpr int META_DIM = 13;
constexpr int ST_DIM = 10;                 // размерность F1 (2, 10, 20, 30, 50, 100)
constexpr long ICO_MAX_CALLS = 50000;
constexpr int CEC_FUNC_ID = 1;             // F1 Shifted and Rotated Bent Cigar

// Целевая функция GWO: запускает ICO с набором параметров x на F1 из CEC2017.
struct ICO_Fitness {
    double user_gray_percent = 0.5;
    int user_genes_bits = 32;
    mutable long calls = 0;

    void reset() const {
        calls = 0;
    }

    double operator()(const Vec& x) const {
        ++calls;

        // BaseAlgorithm::start() рассчитан на известное каноническое решение
        // и останавливается при result.f < f(canonical_x). Для метаоптимизации
        // такого канонического набора ПАРАМЕТРОВ ICO нет. NaN-вектор служит
        // только служебным canonical_x: он получает -infinity, поэтому условие
        // досрочной остановки GWO никогда не сработает.
        const bool gwo_canonical_sentinel =
            x.size() == META_DIM &&
            std::all_of(x.begin(), x.end(), [](double value) {
                return std::isnan(value);
            });
        if (gwo_canonical_sentinel) {
            return -std::numeric_limits<double>::infinity();
        }

        // У каждого оцениваемого набора параметров ICO одинаковая случайная
        // последовательность: сравнение конфигураций честное и воспроизводимое.
        rng_engine.seed(ICO_SEED);

        CountriesAlgorithm::Params params;
        params.p_min = std::min(x[0], x[1]);
        params.p_max = std::max(x[0], x[1]);
        params.M = std::max(2, static_cast<int>(std::round(x[2])));
        params.N = std::max(2, static_cast<int>(std::round(x[3])));
        params.n_min = std::max(1, static_cast<int>(std::round(std::min(x[4], x[5]))));
        params.n_max = std::max(1, static_cast<int>(std::round(std::max(x[4], x[5]))));
        params.m_min = std::max(1, static_cast<int>(std::round(std::min(x[6], x[7]))));
        params.m_max = std::max(1, static_cast<int>(std::round(std::max(x[6], x[7]))));
        params.k = std::max(1, static_cast<int>(std::round(x[8])));
        params.l = std::max(1, static_cast<int>(std::round(x[9])));
        params.ep_elite = x[10];
        params.ep_dead = x[11];
        params.max_mutation = std::max(1, static_cast<int>(std::round(x[12])));

        //params.l = 0;
        //params.ep_elite = x[9];
        //params.ep_dead = x[10];
        //params.max_mutation = std::max(1, static_cast<int>(std::round(x[11])));
        //params.p_war = 0.0;
        //params.p_trade = 1 / 3.0;
        //params.p_motion = 1 /3.0;
        //params.p_epidemic = 1 / 3.0;

        params.p_war = 0.25;
        params.p_trade = 0.25;
        params.p_motion = 0.25;
        params.p_epidemic = 0.25;
        // Диапазон поиска CEC2017 фиксирован: [-100, 100] для всех функций.
        params.x_min = Vec(ST_DIM, cec2017::kLowerBound);
        params.x_max = Vec(ST_DIM, cec2017::kUpperBound);
        params.gray_percent = user_gray_percent;
        params.genes = std::vector<int>(ST_DIM, user_genes_bits);
        params.tmax = 200;
        params.printing = false;

        try {
            // Аналогичный sentinel отключает условие best_f < canonical_y
            // внутри CountriesAlgorithm::start(). Поэтому ICO прекращается
            // именно по лимиту ICO_MAX_CALLS, а не около оптимума F1.
            auto f1_for_ico = [](const Vec& point) -> double {
                const bool ico_canonical_sentinel =
                    point.size() == ST_DIM &&
                    std::all_of(point.begin(), point.end(), [](double value) {
                        return std::isnan(value);
                    });
                if (ico_canonical_sentinel) {
                    return -std::numeric_limits<double>::infinity();
                }
                return cec2017::f1(point);
            };

            CountriesAlgorithm ico(f1_for_ico, std::move(params));
            Vec ico_canonical(ST_DIM, std::numeric_limits<double>::quiet_NaN());
            auto [best_x, best_f, iterations] =
                ico.start(ico_canonical, 0.0, std::nullopt, ICO_MAX_CALLS);
            (void)best_x;
            (void)iterations;
            return best_f;
        } catch (const std::exception&) {
            return std::numeric_limits<double>::infinity();
        }
    }
};

int main() {
    std::cout << "Мета-оптимизация параметров ICO с помощью GWO\n";
    std::cout << "Функция: CEC2017 F1 (Bent Cigar); размерность: " << ST_DIM
              << "; лимит ICO: " << ICO_MAX_CALLS << " вызовов F1\n";
    std::cout << "GWO seed: " << GWO_SEED << "; ICO seed: " << ICO_SEED << "\n\n";

    // p_min, p_max, M, N, n_min, n_max, m_min, m_max, k,
    // ep_elite, ep_dead, max_mutation.
    Vec Xmin = {
        0.000001, 1,
        15.0, 15.0,
        1.0, 2.0,
        1.0, 1.0,
        2.0, 2.0,
        0.01, 0.01,
        16.0
    };

    Vec Xmax = {
        0.5, 2.0,
        50.0, 100.0,
        10.0, 20.0,
        10.0, 15.0,
        10.0, 10.0,
        0.5, 0.5,
        128.0
    };

    ICO_Fitness fitness_function;
    fitness_function.user_gray_percent = 1.0;
    fitness_function.user_genes_bits = 32;

    constexpr int gwo_population_size = 100;
    constexpr int gwo_max_iterations = 200;
    constexpr bool print_gwo = true;

    evo::rng::engine().seed(GWO_SEED);

    evo::GreyWolfOptimizer<ICO_Fitness> gwo(
        fitness_function,
        Xmin,
        Xmax,
        gwo_population_size,
        gwo_max_iterations,
        print_gwo
    );

    // Sentinel отключает некорректный для метаоптимизации критерий
    // result.f < f(canonical_x) в evo::BaseAlgorithm::start().
    Vec gwo_canonical(META_DIM, std::numeric_limits<double>::quiet_NaN());
    auto [best_params, best_fitness, iterations] =
        gwo.start(gwo_canonical, 0.0, std::nullopt, std::nullopt);

    // Известный оптимум F1 в CEC2017: f(x*) = funcid * 100.
    const double true_optimum = cec2017::optimum_value(CEC_FUNC_ID);
    const double best_error = best_fitness - true_optimum;

    std::cout << "\n=== Оптимизация завершена ===\n";
    std::cout << "Итераций GWO: " << iterations << "\n";
    std::cout << "Лучшее найденное значение F1: " << best_fitness << "\n";
    std::cout << "Известный оптимум F1: " << true_optimum << "\n";
    std::cout << "Ошибка (error = best_fitness - optimum): " << best_error << "\n\n";
    std::cout << "Найденные параметры ICO:\n";
    std::cout << "p_min: " << std::min(best_params[0], best_params[1]) << "\n";
    std::cout << "p_max: " << std::max(best_params[0], best_params[1]) << "\n";
    std::cout << "M: " << std::max(2, static_cast<int>(std::round(best_params[2]))) << "\n";
    std::cout << "N: " << std::max(2, static_cast<int>(std::round(best_params[3]))) << "\n";
    std::cout << "n_min: " << std::max(1, static_cast<int>(std::round(std::min(best_params[4], best_params[5])))) << "\n";
    std::cout << "n_max: " << std::max(1, static_cast<int>(std::round(std::max(best_params[4], best_params[5])))) << "\n";
    std::cout << "m_min: " << std::max(1, static_cast<int>(std::round(std::min(best_params[6], best_params[7])))) << "\n";
    std::cout << "m_max: " << std::max(1, static_cast<int>(std::round(std::max(best_params[6], best_params[7])))) << "\n";
    std::cout << "k: " << std::max(1, static_cast<int>(std::round(best_params[8]))) << "\n";
    std::cout << "l: " << std::max(1, static_cast<int>(std::round(best_params[9]))) << "\n";
    // std::cout << "l: " << std::max(1, static_cast<int>(std::round(best_params[9]))) << "\n";
    std::cout << "ep_elite: " << best_params[10] << "\n";
    std::cout << "ep_dead: " << best_params[11] << "\n";
    std::cout << "max_mutation: " << std::max(1, static_cast<int>(std::round(best_params[12]))) << "\n";
    // std::cout << "p_war: 0 (фиксировано)\n";
    // std::cout << "p_trade: 0.33 (фиксировано)\n";
    //                                                              std::cout << "p_motion: 0.33 (фиксировано)\n";
    //std::cout << "p_epidemic: 0.33 (фиксировано)\n";

    return 0;
}
