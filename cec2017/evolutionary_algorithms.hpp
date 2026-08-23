#pragma once
// ============================================================================
// evolutionary_algorithms.hpp
// C++17 port of evolutionary_algorithms.py
//
// Реализация эволюционных алгоритмов / метаэвристик, совместимых по интерфейсу
// с BaseBenchMark из benchmark.hpp (метод start(x0, x_eps, y_eps, max_calls)).
//
// Каждый алгоритм:
//   - принимает функцию f (CountedFunction<...> с полями .calls и методом .reset()),
//     границы Xmin, Xmax (Vec), размер популяции N, число итераций tmax;
//   - имеет метод start(x0, x_eps, y_eps, max_calls) -> tuple<Vec, double, long>,
//     совместимый с контрактом BaseBenchMark<Method, TestFunc>::operator()();
//   - хранит текущее лучшее решение в result_ (Solution: real_x + f).
//
// Источники (описание алгоритмов):
//   GA   - Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). "A Fast and
//          Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE Transactions
//          on Evolutionary Computation, 6(2), 182-197.
//   ABC  - Karaboga, D. (2005). "An idea based on honey bee swarm for numerical
//          optimization." Technical Report TR06, Erciyes University.
//   IWO  - Mehrabian, A.R., Lucas, C. (2006). "A novel numerical optimization
//          algorithm inspired from weed colonization." Ecological Informatics, 1(4), 355-366.
//   SCSO - Seyyedabbasi, A., Kiani, F. (2022). "Sand Cat Swarm Optimization: a
//          nature-inspired algorithm to solve global optimization problems."
//          Engineering with Computers, 39, 2627-2651.
//   GWO  - Mirjalili, S., Mirjalili, S.M., Lewis, A. (2014). "Grey Wolf Optimizer."
//          Advances in Engineering Software, 69, 46-61.
// ============================================================================

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <optional>
#include <tuple>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace evo {

using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
// RNG — единая точка для генерации случайных чисел (аналог np.random / random)
// ---------------------------------------------------------------------------
namespace rng {
inline std::mt19937& engine() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}
inline double uniform(double a = 0.0, double b = 1.0) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(engine());
}
inline double normal(double mean = 0.0, double sigma = 1.0) {
    std::normal_distribution<double> dist(mean, sigma);
    return dist(engine());
}
inline int randint(int lo, int hi_inclusive) {
    std::uniform_int_distribution<int> dist(lo, hi_inclusive);
    return dist(engine());
}
// Взвешенный выбор индекса по вектору вероятностей (аналог np.random.choice(p=...))
inline int choice_weighted(const Vec& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(engine());
}
} // namespace rng

// ---------------------------------------------------------------------------
// Вспомогательные векторные операции
// ---------------------------------------------------------------------------
inline Vec vec_clip(const Vec& x, const Vec& lo, const Vec& hi) {
    Vec r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = std::min(std::max(x[i], lo[i]), hi[i]);
    return r;
}
inline Vec vec_uniform(const Vec& lo, const Vec& hi) {
    Vec r(lo.size());
    for (size_t i = 0; i < lo.size(); ++i) r[i] = rng::uniform(lo[i], hi[i]);
    return r;
}
inline Vec vec_add(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}
inline Vec vec_sub(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}
inline Vec vec_scale(const Vec& a, double s) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}
inline double vec_norm_diff(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

// ---------------------------------------------------------------------------
// Solution — аналог Solution (real_x, f)
// ---------------------------------------------------------------------------
struct Solution {
    Vec real_x;
    double f = std::numeric_limits<double>::infinity();

    Solution() = default;
    Solution(Vec x, double fv) : real_x(std::move(x)), f(fv) {}

    bool operator<(const Solution& o) const { return f < o.f; }
};

// ---------------------------------------------------------------------------
// BaseAlgorithm<TestFunc> — общий каркас (аналог BaseAlgorithm)
//
// TestFunc должен поддерживать:
//   double operator()(const Vec&) const   — вычисление функции
//   long calls (mutable)                  — счётчик вызовов
//   void reset() const                    — сброс счётчика
// Именно такой интерфейс предоставляет CountedFunction<F> из test_functions.hpp
// ---------------------------------------------------------------------------
template <typename TestFunc>
class BaseAlgorithm {
public:
    BaseAlgorithm(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax, bool printing = false)
        : f_(f), Xmin_(std::move(Xmin)), Xmax_(std::move(Xmax)),
          dim_(Xmin_.size()), N_(N), tmax_(tmax), printing_(printing), ti_(0) {}

    virtual ~BaseAlgorithm() = default;

    // Аналог start() без аргументов -> (best_x, best_f, False, ti)
    std::tuple<Vec, double, bool, long> start_free() {
        first_iteration();
        while (ti_ <= tmax_) {
            bool stop = iteration();
            if (stop) break;
        }
        return {result_.real_x, result_.f, false, static_cast<long>(ti_)};
    }

    // Аналог testing(canonical_x, epsilon, y_epsilon, max_calls) -> (best_x, best_f, ti)
    // Сигнатура совместима с контрактом Method::start(...) из benchmark.hpp
    std::tuple<Vec, double, long> start(const Vec& canonical_x, double epsilon,
                                         std::optional<double> y_epsilon,
                                         std::optional<long> max_calls) {
        double canonical_y = f_(canonical_x);
        first_iteration();

        while (ti_ <= tmax_) {
            if (max_calls.has_value() && f_.calls >= *max_calls) {
                if (printing_) {
                    std::cout << result_.f << " ti=" << ti_
                              << " norm=" << vec_norm_diff(canonical_x, result_.real_x) << std::endl;
                    std::cout << "Лимит вызовов функции достигнут" << std::endl;
                }
                return {result_.real_x, result_.f, static_cast<long>(ti_)};
            }

            bool stop = iteration();
            if (stop) break;

            if (y_epsilon.has_value() && std::fabs(result_.f - canonical_y) <= *y_epsilon) {
                if (printing_) {
                    std::cout << result_.f << " ti=" << ti_
                              << " norm=" << vec_norm_diff(canonical_x, result_.real_x) << std::endl;
                }
                return {result_.real_x, result_.f, static_cast<long>(ti_)};
            }


            if (result_.f < canonical_y) {
                if (printing_) {
                    std::cout << result_.f << " ti=" << ti_
                              << " norm=" << vec_norm_diff(canonical_x, result_.real_x) << std::endl;
                }
                return {result_.real_x, result_.f, static_cast<long>(ti_)};
            }

            if (max_calls.has_value() && f_.calls >= *max_calls) {
                if (printing_) {
                    std::cout << result_.f << " ti=" << ti_
                              << " norm=" << vec_norm_diff(canonical_x, result_.real_x) << std::endl;
                    std::cout << "Лимит вызовов функции достигнут " << f_.calls << std::endl;
                }
                return {result_.real_x, result_.f, static_cast<long>(ti_)};
            }

            if (printing_ && ti_ % 250 == 0) {
                std::cout << "Current result: f=" << result_.f << " ti=" << ti_ << std::endl;
            }
        }

        if (printing_) {
            std::cout << result_.f << " ti=" << ti_
                      << " norm=" << vec_norm_diff(canonical_x, result_.real_x) << std::endl;
        }
        return {result_.real_x, result_.f, static_cast<long>(ti_)};
    }

protected:
    double eval(const Vec& x) { return f_(x); }
    Vec clip(const Vec& x) const { return vec_clip(x, Xmin_, Xmax_); }

    std::pair<std::vector<Vec>, Vec> init_population() {
        std::vector<Vec> pop_x(N_);
        Vec pop_f(N_);
        for (int i = 0; i < N_; ++i) {
            pop_x[i] = vec_uniform(Xmin_, Xmax_);
            pop_f[i] = eval(pop_x[i]);
        }
        return {pop_x, pop_f};
    }

    void update_result(const std::vector<Vec>& pop_x, const Vec& pop_f) {
        size_t idx = std::distance(pop_f.begin(), std::min_element(pop_f.begin(), pop_f.end()));
        Solution candidate(pop_x[idx], pop_f[idx]);
        if (!has_result_ || candidate.f < result_.f) {
            result_ = candidate;
            has_result_ = true;
        }
    }

    virtual void first_iteration() {
        ti_ = 0;
        std::tie(pop_x_, pop_f_) = init_population();
        has_result_ = false;
        update_result(pop_x_, pop_f_);
    }

    // Должен быть переопределён в дочернем классе. true при досрочной остановке.
    virtual bool iteration() = 0;

    TestFunc& f_;
    Vec Xmin_, Xmax_;
    size_t dim_;
    int N_, tmax_;
    bool printing_;
    int ti_;

    std::vector<Vec> pop_x_;
    Vec pop_f_;
    Solution result_;
    bool has_result_ = false;
};

// ===========================================================================
// 1. Genetic Algorithm (GA) — Deb et al., 2002 (NSGA-II style real-coded GA)
// Турнирная селекция, SBX-кроссовер, гауссова мутация, элитизм.
// ===========================================================================
template <typename TestFunc>
class GeneticAlgorithm : public BaseAlgorithm<TestFunc> {
    using Base = BaseAlgorithm<TestFunc>;
    using Base::dim_; using Base::N_; using Base::Xmin_; using Base::Xmax_;
    using Base::pop_x_; using Base::pop_f_; using Base::eval; using Base::clip;
    using Base::update_result; using Base::ti_; using Base::printing_; using Base::result_;

public:
    GeneticAlgorithm(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax,
                      double crossover_rate = 0.9, double mutation_rate = 0.1, int elite = 2,
                      int tournament_k = 3, double sbx_eta = 15.0, bool printing = false)
        : Base(f, std::move(Xmin), std::move(Xmax), N, tmax, printing),
          crossover_rate_(crossover_rate), mutation_rate_(mutation_rate), elite_(elite),
          tournament_k_(tournament_k), sbx_eta_(sbx_eta) {}

protected:
    Vec tournament_select() {
        int best = -1; double best_f = std::numeric_limits<double>::infinity();
        for (int k = 0; k < tournament_k_; ++k) {
            int idx = rng::randint(0, N_ - 1);
            if (pop_f_[idx] < best_f) { best_f = pop_f_[idx]; best = idx; }
        }
        return pop_x_[best];
    }

    std::pair<Vec, Vec> sbx(const Vec& p1, const Vec& p2) {
        Vec c1(dim_), c2(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            double u = rng::uniform();
            double beta = (u <= 0.5)
                ? std::pow(2.0 * u, 1.0 / (sbx_eta_ + 1.0))
                : std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (sbx_eta_ + 1.0));
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i]);
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i]);
        }
        return {clip(c1), clip(c2)};
    }

    Vec mutate(Vec x) {
        for (size_t i = 0; i < dim_; ++i) {
            if (rng::uniform() < mutation_rate_) {
                double sigma = 0.1 * (Xmax_[i] - Xmin_[i]);
                x[i] += rng::normal(0.0, 1.0) * sigma;
            }
        }
        return clip(x);
    }

    bool iteration() override {
        ti_ += 1;
        std::vector<int> order(N_);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return pop_f_[a] < pop_f_[b]; });

        std::vector<Vec> elite_x;
        Vec elite_f;
        for (int i = 0; i < elite_; ++i) { elite_x.push_back(pop_x_[order[i]]); elite_f.push_back(pop_f_[order[i]]); }

        std::vector<Vec> new_x;
        while (static_cast<int>(new_x.size()) < N_ - elite_) {
            Vec p1 = tournament_select();
            Vec p2 = tournament_select();
            Vec c1, c2;
            if (rng::uniform() < crossover_rate_) {
                std::tie(c1, c2) = sbx(p1, p2);
            } else {
                c1 = p1; c2 = p2;
            }
            new_x.push_back(mutate(c1));
            new_x.push_back(mutate(c2));
        }
        new_x.resize(N_ - elite_);
        Vec new_f(new_x.size());
        for (size_t i = 0; i < new_x.size(); ++i) new_f[i] = eval(new_x[i]);

        pop_x_ = elite_x; pop_x_.insert(pop_x_.end(), new_x.begin(), new_x.end());
        pop_f_ = elite_f; pop_f_.insert(pop_f_.end(), new_f.begin(), new_f.end());
        update_result(pop_x_, pop_f_);

        if (printing_) std::cout << ti_ << ") GA best: " << result_.f << std::endl;
        return false;
    }

private:
    double crossover_rate_, mutation_rate_;
    int elite_, tournament_k_;
    double sbx_eta_;
};

// ===========================================================================
// 2. Artificial Bee Colony (ABC) — Karaboga, 2005
// employed / onlooker / scout фазы
// ===========================================================================
template <typename TestFunc>
class ArtificialBeeColony : public BaseAlgorithm<TestFunc> {
    using Base = BaseAlgorithm<TestFunc>;
    using Base::dim_; using Base::N_; using Base::pop_x_; using Base::pop_f_;
    using Base::eval; using Base::clip; using Base::update_result;
    using Base::ti_; using Base::printing_; using Base::result_;

public:
    ArtificialBeeColony(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax,
                         std::optional<double> limit = std::nullopt, bool printing = false)
        : Base(f, std::move(Xmin), std::move(Xmax), N, tmax, printing) {
        limit_ = limit.has_value() ? *limit : static_cast<double>(N) * static_cast<double>(dim_);
    }

protected:
    void first_iteration() override {
        Base::first_iteration();
        trial_.assign(N_, 0.0);
    }

    static double fitness_one(double fv) {
        return (fv >= 0) ? 1.0 / (1.0 + fv) : 1.0 + std::fabs(fv);
    }

    Vec new_candidate(int i) {
        int k = i;
        while (k == i) k = rng::randint(0, N_ - 1);
        Vec candidate(dim_);
        for (size_t d = 0; d < dim_; ++d) {
            double phi = rng::uniform(-1.0, 1.0);
            candidate[d] = pop_x_[i][d] + phi * (pop_x_[i][d] - pop_x_[k][d]);
        }
        return clip(candidate);
    }

    void employed_phase() {
        for (int i = 0; i < N_; ++i) {
            Vec candidate = new_candidate(i);
            double cand_f = eval(candidate);
            if (cand_f < pop_f_[i]) { pop_x_[i] = candidate; pop_f_[i] = cand_f; trial_[i] = 0; }
            else trial_[i] += 1;
        }
    }

    void onlooker_phase() {
        Vec fit(N_);
        double sum = 0.0;
        for (int i = 0; i < N_; ++i) { fit[i] = fitness_one(pop_f_[i]); sum += fit[i]; }
        Vec prob(N_);
        for (int i = 0; i < N_; ++i) prob[i] = fit[i] / sum;

        for (int j = 0; j < N_; ++j) {
            int i = rng::choice_weighted(prob);
            Vec candidate = new_candidate(i);
            double cand_f = eval(candidate);
            if (cand_f < pop_f_[i]) { pop_x_[i] = candidate; pop_f_[i] = cand_f; trial_[i] = 0; }
            else trial_[i] += 1;
        }
    }

    void scout_phase() {
        for (int i = 0; i < N_; ++i) {
            if (trial_[i] > limit_) {
                pop_x_[i] = vec_uniform(this->Xmin_, this->Xmax_);
                pop_f_[i] = eval(pop_x_[i]);
                trial_[i] = 0;
            }
        }
    }

    bool iteration() override {
        ti_ += 1;
        employed_phase();
        onlooker_phase();
        scout_phase();
        update_result(pop_x_, pop_f_);

        if (printing_) std::cout << ti_ << ") ABC best: " << result_.f << std::endl;
        return false;
    }

private:
    double limit_;
    Vec trial_;
};

// ===========================================================================
// 3. Invasive Weed Optimization (IWO) — Mehrabian & Lucas, 2006
// ===========================================================================
template <typename TestFunc>
class InvasiveWeedOptimization : public BaseAlgorithm<TestFunc> {
    using Base = BaseAlgorithm<TestFunc>;
    using Base::dim_; using Base::N_; using Base::Xmin_; using Base::Xmax_;
    using Base::pop_x_; using Base::pop_f_; using Base::eval; using Base::clip;
    using Base::update_result; using Base::ti_; using Base::tmax_;
    using Base::printing_; using Base::result_;

public:
    // sigma_initial_frac / sigma_final_frac задаются как ДОЛИ от области поиска
    // по каждой координате: sigma[i] = frac * (Xmax[i] - Xmin[i]).
    // Можно передать:
    //   - std::nullopt               -> используется значение по умолчанию
    //   - Vec из одного элемента     -> одинаковая доля по всем координатам
    //   - Vec размера dim            -> своя доля для каждой координаты
    // Значения по умолчанию (канонические для IWO, Mehrabian & Lucas, 2006):
    //   sigma_initial_frac = 0.5   (широкий начальный разброс семян)
    //   sigma_final_frac   = 1e-6  (узкий финальный разброс для точной сходимости;
    //                               исходное значение 0.001 давало слишком грубую
    //                               финальную дисперсию и мешало сходимости на поздних итерациях)
    static constexpr double kDefaultSigmaInitialFrac = 0.5;
    static constexpr double kDefaultSigmaFinalFrac = 1e-6;

    InvasiveWeedOptimization(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax,
                              int n_min = 1, int n_max = 5, std::optional<int> s_max = std::nullopt,
                              std::optional<Vec> sigma_initial_frac = std::nullopt,
                              std::optional<Vec> sigma_final_frac = std::nullopt,
                              double m_exp = 3.0, bool printing = false)
        : Base(f, std::move(Xmin), std::move(Xmax), N, tmax, printing),
          n_min_(n_min), n_max_(n_max), m_exp_(m_exp) {
        s_max_ = s_max.has_value() ? *s_max : 2 * N;
        sigma_initial_ = fractions_to_sigma(sigma_initial_frac, kDefaultSigmaInitialFrac);
        sigma_final_ = fractions_to_sigma(sigma_final_frac, kDefaultSigmaFinalFrac);
    }

    // Удобный конструктор: одна и та же доля по всем координатам (double вместо Vec).
    InvasiveWeedOptimization(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax,
                              int n_min, int n_max, std::optional<int> s_max,
                              double sigma_initial_frac,
                              double sigma_final_frac,
                              double m_exp = 3.0, bool printing = false)
        : InvasiveWeedOptimization(f, std::move(Xmin), std::move(Xmax), N, tmax,
                                    n_min, n_max, s_max,
                                    Vec(1, sigma_initial_frac),
                                    Vec(1, sigma_final_frac),
                                    m_exp, printing) {}

protected:
    // Преобразует долю(и) от области поиска в абсолютный вектор sigma по каждой координате.
    Vec fractions_to_sigma(const std::optional<Vec>& frac_opt, double default_frac) const {
        Vec s(dim_);
        if (frac_opt.has_value()) {
            const Vec& frac = *frac_opt;
            for (size_t i = 0; i < dim_; ++i) {
                double fr = (frac.size() == 1) ? frac[0] : frac[i];
                s[i] = fr * (Xmax_[i] - Xmin_[i]);
            }
        } else {
            for (size_t i = 0; i < dim_; ++i) s[i] = default_frac * (Xmax_[i] - Xmin_[i]);
        }
        return s;
    }

    Vec sigma() const {
        double ratio = std::pow(static_cast<double>(std::max(0, tmax_ - ti_)) / tmax_, m_exp_);
        Vec s(dim_);
        for (size_t i = 0; i < dim_; ++i) s[i] = sigma_final_[i] + (sigma_initial_[i] - sigma_final_[i]) * ratio;
        return s;
    }

    bool iteration() override {
        ti_ += 1;
        double f_min = *std::min_element(pop_f_.begin(), pop_f_.end());
        double f_max = *std::max_element(pop_f_.begin(), pop_f_.end());
        Vec sig = sigma();

        std::vector<Vec> new_x_list;
        Vec new_f_list;

        for (size_t i = 0; i < pop_x_.size(); ++i) {
            int n_seeds;
            if (f_max == f_min) n_seeds = n_min_;
            else {
                double ratio = (f_max - pop_f_[i]) / (f_max - f_min);
                n_seeds = static_cast<int>(std::lround(n_min_ + ratio * (n_max_ - n_min_)));
            }
            for (int s = 0; s < n_seeds; ++s) {
                Vec seed(dim_);
                for (size_t d = 0; d < dim_; ++d) seed[d] = pop_x_[i][d] + sig[d] * rng::normal(0.0, 1.0);
                seed = clip(seed);
                new_x_list.push_back(seed);
                new_f_list.push_back(eval(seed));
            }
        }

        std::vector<Vec> all_x = pop_x_;
        Vec all_f = pop_f_;
        all_x.insert(all_x.end(), new_x_list.begin(), new_x_list.end());
        all_f.insert(all_f.end(), new_f_list.begin(), new_f_list.end());

        std::vector<int> order(all_x.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return all_f[a] < all_f[b]; });
        int keep = std::min(static_cast<int>(order.size()), s_max_);

        std::vector<Vec> new_pop_x(keep);
        Vec new_pop_f(keep);
        for (int i = 0; i < keep; ++i) { new_pop_x[i] = all_x[order[i]]; new_pop_f[i] = all_f[order[i]]; }
        pop_x_ = new_pop_x; pop_f_ = new_pop_f;
        update_result(pop_x_, pop_f_);

        if (printing_) std::cout << ti_ << ") IWO best: " << result_.f
                                  << ", размер колонии: " << pop_x_.size() << std::endl;
        return false;
    }

private:
    int n_min_, n_max_, s_max_;
    Vec sigma_initial_, sigma_final_;
    double m_exp_;
};

// ===========================================================================
// 4. Sand Cat Swarm Optimization (SCSO) — Seyyedabbasi & Kiani, 2022
// ===========================================================================
template <typename TestFunc>
class SandCatSwarmOptimization : public BaseAlgorithm<TestFunc> {
    using Base = BaseAlgorithm<TestFunc>;
    using Base::dim_; using Base::N_; using Base::pop_x_; using Base::pop_f_;
    using Base::eval; using Base::clip; using Base::update_result;
    using Base::ti_; using Base::tmax_; using Base::printing_; using Base::result_;

public:
    SandCatSwarmOptimization(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax,
                              double s_m = 2.0, bool printing = false)
        : Base(f, std::move(Xmin), std::move(Xmax), N, tmax, printing), s_m_(s_m) {}

protected:
    bool iteration() override {
        ti_ += 1;
        double rg = s_m_ - ti_ * (s_m_ / tmax_);
        const Vec& best = result_.real_x;

        for (int i = 0; i < N_; ++i) {
            double r = rg * rng::uniform();
            double R = 2 * rg * rng::uniform() - rg;
            double theta = rng::uniform(0.0, 2.0 * M_PI);

            Vec candidate(dim_);
            if (rng::uniform() < 0.5) {
                for (size_t d = 0; d < dim_; ++d)
                    candidate[d] = best[d] - r * pop_x_[i][d] * std::cos(theta);
            } else {
                for (size_t d = 0; d < dim_; ++d)
                    candidate[d] = best[d] - R * pop_x_[i][d];
            }

            candidate = clip(candidate);
            double cand_f = eval(candidate);
            if (cand_f < pop_f_[i]) { pop_x_[i] = candidate; pop_f_[i] = cand_f; }
        }

        update_result(pop_x_, pop_f_);

        if (printing_) std::cout << ti_ << ") SCSO best: " << result_.f << std::endl;
        return false;
    }

private:
    double s_m_;
};

// ===========================================================================
// 5. Grey Wolf Optimizer (GWO) — Mirjalili et al., 2014
// ===========================================================================
template <typename TestFunc>
class GreyWolfOptimizer : public BaseAlgorithm<TestFunc> {
    using Base = BaseAlgorithm<TestFunc>;
    using Base::dim_; using Base::N_; using Base::pop_x_; using Base::pop_f_;
    using Base::eval; using Base::clip; using Base::update_result;
    using Base::ti_; using Base::tmax_; using Base::printing_; using Base::result_;

public:
    GreyWolfOptimizer(TestFunc& f, Vec Xmin, Vec Xmax, int N, int tmax, bool printing = false)
        : Base(f, std::move(Xmin), std::move(Xmax), N, tmax, printing) {}

protected:
    std::tuple<Vec, Vec, Vec> leaders() const {
        std::vector<int> order(N_);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 3, order.end(),
                           [&](int a, int b) { return pop_f_[a] < pop_f_[b]; });
        return {pop_x_[order[0]], pop_x_[order[1]], pop_x_[order[2]]};
    }

    bool iteration() override {
        ti_ += 1;
        double a = 2.0 - ti_ * (2.0 / tmax_);
        auto [alpha, beta, delta] = leaders();

        std::vector<Vec> new_x(N_, Vec(dim_));
        for (int i = 0; i < N_; ++i) {
            Vec sum(dim_, 0.0);
            for (const Vec* leader : {&alpha, &beta, &delta}) {
                Vec pos(dim_);
                for (size_t d = 0; d < dim_; ++d) {
                    double r1 = rng::uniform(), r2 = rng::uniform();
                    double A = 2 * a * r1 - a;
                    double C = 2 * r2;
                    double D = std::fabs(C * (*leader)[d] - pop_x_[i][d]);
                    pos[d] = (*leader)[d] - A * D;
                }
                for (size_t d = 0; d < dim_; ++d) sum[d] += pos[d];
            }
            for (size_t d = 0; d < dim_; ++d) sum[d] /= 3.0;
            new_x[i] = clip(sum);
        }

        Vec new_f(N_);
        for (int i = 0; i < N_; ++i) new_f[i] = eval(new_x[i]);
        pop_x_ = new_x; pop_f_ = new_f;
        update_result(pop_x_, pop_f_);

        if (printing_) std::cout << ti_ << ") GWO best: " << result_.f << std::endl;
        return false;
    }
};

} // namespace evo
