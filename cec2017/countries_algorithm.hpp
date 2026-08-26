// ============================================================
// countries_algorithm.hpp
//
// Адаптация ico.cpp для автономной работы без pybind11/Python.
// С добавленным методом start(canonical_x, epsilon, y_epsilon, max_calls)
// для тестирования и сравнения с Python-версией.
// СЧЕТЧИК ИСПРАВЛЕН (используется shared_ptr<long>).
// ============================================================
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <optional>

// ============================================================
//  Gray code utilities
// ============================================================

static inline uint64_t tc_to_gray_code(uint64_t n) noexcept {
    return n ^ (n >> 1);
}

static inline uint64_t gray_code_to_tc(uint64_t g) noexcept {
    uint64_t n = 0;
    for (; g; g >>= 1) n ^= g;
    return n;
}

static thread_local std::mt19937_64 rng_engine{std::random_device{}()};

static inline void set_random_seed(std::uint64_t seed) {
    rng_engine.seed(seed);
}

static inline double rand_uniform(double lo, double hi) {
    return std::uniform_real_distribution<double>{lo, hi}(rng_engine);
}

static inline int rand_int(int lo, int hi_inclusive) {
    return std::uniform_int_distribution<int>{lo, hi_inclusive}(rng_engine);
}

static inline uint64_t rand_uint64(uint64_t lo, uint64_t hi_inclusive) {
    return std::uniform_int_distribution<uint64_t>{lo, hi_inclusive}(rng_engine);
}

// Взвешенный случайный выбор действия по заданным вероятностям.
// probs.size() == 4: [Война, Обмен, Движение к лидеру, Эпидемия]
// Возвращает индекс действия в порядке, совместимом с action-кодами:
// 0 = Движение (Motion), 1 = Обмен (Trade), 2 = Война (War), 3 = Эпидемия (Epidemic)
static inline int weighted_action_choice(double p_war, double p_trade,
                                          double p_motion, double p_epidemic) {
    // Порядок массива вероятностей соответствует action-кодам: motion, trade, war, epidemic
    double probs[4] = { p_motion, p_trade, p_war, p_epidemic };
    double sum = probs[0] + probs[1] + probs[2] + probs[3];
    if (sum <= 0.0) return rand_int(0, 3);
    double r = rand_uniform(0.0, sum);
    double acc = 0.0;
    for (int i = 0; i < 4; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return 3;
}

enum class IndividualType { Gray, Real };

using FuncT = std::function<double(const std::vector<double>&)>;

struct Individual {
    std::vector<double> x_min, x_max;
    double f_value = std::numeric_limits<double>::infinity();
    int    ep_n    = 0;
    IndividualType itype;

    Individual(std::vector<double> xmin, std::vector<double> xmax, IndividualType t)
        : x_min(std::move(xmin)), x_max(std::move(xmax)), itype(t) {}

    virtual ~Individual() = default;
    virtual std::vector<double> real_x() const = 0;
    virtual std::shared_ptr<Individual> clone() const = 0;

    bool operator<(const Individual& o) const noexcept { return f_value < o.f_value; }
    bool operator>(const Individual& o) const noexcept { return f_value > o.f_value; }
    bool operator==(const Individual& o) const noexcept { return f_value == o.f_value; }
};

struct GrayIndividual : Individual {
    std::vector<int>      genes;
    std::vector<uint64_t> code;
    std::vector<double>   steps;

    GrayIndividual(
        std::vector<uint64_t> gray_code,
        const std::vector<double>& xmin,
        const std::vector<double>& xmax,
        const std::vector<int>&    genes_,
        const FuncT& func)
        : Individual(xmin, xmax, IndividualType::Gray),
          genes(genes_), code(std::move(gray_code))
    {
        _init_steps();
        f_value = func(real_x());
    }

    GrayIndividual(
        std::vector<uint64_t> gray_code,
        const std::vector<double>& xmin,
        const std::vector<double>& xmax,
        const std::vector<int>&    genes_,
        double f_val)
        : Individual(xmin, xmax, IndividualType::Gray),
          genes(genes_), code(std::move(gray_code))
    {
        _init_steps();
        f_value = f_val;
    }

    std::vector<double> real_x() const override {
        int dim = (int)genes.size();
        std::vector<double> rx(dim);
        for (int i = 0; i < dim; ++i)
            rx[i] = x_min[i] + steps[i] * (double)gray_code_to_tc(code[i]);
        return rx;
    }

    std::vector<uint64_t> decimal_x() const {
        int dim = (int)genes.size();
        std::vector<uint64_t> dec(dim);
        for (int i = 0; i < dim; ++i)
            dec[i] = gray_code_to_tc(code[i]);
        return dec;
    }

    static std::shared_ptr<GrayIndividual> from_decimal(
        const std::vector<uint64_t>& decimal,
        const std::vector<double>&   xmin,
        const std::vector<double>&   xmax,
        const std::vector<int>&      genes_,
        const FuncT& func)
    {
        int dim = (int)genes_.size();
        std::vector<uint64_t> gc(dim);
        for (int i = 0; i < dim; ++i) {
            uint64_t maxval = (1ULL << genes_[i]) - 1;
            gc[i] = tc_to_gray_code(std::min(decimal[i], maxval));
        }
        return std::make_shared<GrayIndividual>(gc, xmin, xmax, genes_, func);
    }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
        p->ep_n = ep_n;
        return p;
    }

    std::shared_ptr<GrayIndividual> mutate(double q_max_term, const FuncT& func) const {
        // формула (7): q_c = max(0, (1 - t/tmax)*q_max - n_ep)
        int n = std::max(0, (int)std::floor(q_max_term) - ep_n);

        if (n == 0) {
            // без мутации — переиспользуем уже известный f_value, без нового вызова func
            auto p = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
            return p;
        }

        int total_bits = 0;
        for (int g : genes) total_bits += g;

        std::vector<uint8_t> bits(total_bits);
        {
            int pos = 0;
            for (int i = 0; i < (int)genes.size(); ++i)
                for (int b = genes[i] - 1; b >= 0; --b)
                    bits[pos++] = (uint8_t)((code[i] >> b) & 1);
        }
        std::vector<int> positions(total_bits);
        std::iota(positions.begin(), positions.end(), 0);
        int flips = std::min(n, total_bits);
        for (int i = 0; i < flips; ++i) {
            int j = rand_int(i, total_bits - 1);
            std::swap(positions[i], positions[j]);
            bits[positions[i]] ^= 1;
        }
        std::vector<uint64_t> new_code(genes.size(), 0);
        {
            int pos = 0;
            for (int i = 0; i < (int)genes.size(); ++i) {
                uint64_t v = 0;
                for (int b = genes[i] - 1; b >= 0; --b)
                    v |= ((uint64_t)bits[pos++] << b);
                new_code[i] = v;
            }
        }
        auto p = std::make_shared<GrayIndividual>(new_code, x_min, x_max, genes, func);
        return p;
    }

    // static std::pair<std::shared_ptr<GrayIndividual>, std::shared_ptr<GrayIndividual>>
    // crossover(const GrayIndividual& a, const GrayIndividual& b, const FuncT& func) {
    //     int total_bits = 0;
    //     for (int g : a.genes) total_bits += g;
    //
    //     auto to_bits = [&](const GrayIndividual& ind) {
    //         std::vector<uint8_t> bits(total_bits);
    //         int pos = 0;
    //         for (int i = 0; i < (int)ind.genes.size(); ++i)
    //             for (int bit = ind.genes[i] - 1; bit >= 0; --bit)
    //                 bits[pos++] = (uint8_t)((ind.code[i] >> bit) & 1);
    //         return bits;
    //     };
    //
    //     auto bits_a = to_bits(a);
    //     auto bits_b = to_bits(b);
    //
    //     int k1, k2;
    //     if (total_bits <= 2) {
    //         k1 = 0;
    //         k2 = total_bits;
    //     } else {
    //         k1 = rand_int(1, total_bits - 2);
    //         k2 = rand_int(1, total_bits - 2);
    //
    //         while (k2 == k1)
    //             k2 = rand_int(1, total_bits - 2);
    //
    //         if (k1 > k2)
    //             std::swap(k1, k2);
    //     }
    //
    //     std::vector<uint8_t> nb1(total_bits), nb2(total_bits);
    //     for (int i = 0; i < total_bits; ++i) {
    //         nb1[i] = (i >= k1 && i < k2) ? bits_b[i] : bits_a[i];
    //         nb2[i] = (i >= k1 && i < k2) ? bits_a[i] : bits_b[i];
    //     }
    //
    //     auto bits_to_ind = [&](const std::vector<uint8_t>& bits) {
    //         std::vector<uint64_t> gc(a.genes.size(), 0);
    //         int pos = 0;
    //
    //         for (int i = 0; i < (int)a.genes.size(); ++i) {
    //             uint64_t v = 0;
    //             for (int bit = a.genes[i] - 1; bit >= 0; --bit)
    //                 v |= ((uint64_t)bits[pos++] << bit);
    //             gc[i] = v;
    //         }
    //
    //         return std::make_shared<GrayIndividual>(
    //             gc, a.x_min, a.x_max, a.genes, func
    //         );
    //     };
    //
    //     return {bits_to_ind(nb1), bits_to_ind(nb2)};
    // }

        static std::pair<std::shared_ptr<GrayIndividual>, std::shared_ptr<GrayIndividual>>
    crossover(const GrayIndividual& a, const GrayIndividual& b, const FuncT& func) {
        int total_bits = 0;
        for (int g : a.genes) total_bits += g;

        auto to_bits = [&](const GrayIndividual& ind) {
            std::vector<uint8_t> bits(total_bits);
            int pos = 0;
            for (int i = 0; i < (int)ind.genes.size(); ++i)
                for (int bit = ind.genes[i] - 1; bit >= 0; --bit)
                    bits[pos++] = (uint8_t)((ind.code[i] >> bit) & 1);
            return bits;
        };

        auto bits_a = to_bits(a);
        auto bits_b = to_bits(b);

        std::vector<uint8_t> nb1(total_bits), nb2(total_bits);

        // В 70% случаев используем Uniform Crossover, в 30% - двухточечный
        if (rand_uniform(0.0, 1.0) < 0.7) {
            // Uniform crossover
            for (int i = 0; i < total_bits; ++i) {
                // С вероятностью 50% меняем биты родителей местами для потомков
                if (rand_int(0, 1) == 1) {
                    nb1[i] = bits_b[i];
                    nb2[i] = bits_a[i];
                } else {
                    nb1[i] = bits_a[i];
                    nb2[i] = bits_b[i];
                }
            }
        } else {
            // Двухточечный кроссовер (исходная логика)
            int k1, k2;
            if (total_bits <= 2) {
                k1 = 0; k2 = total_bits;
            } else {
                k1 = rand_int(1, total_bits - 2);
                k2 = rand_int(1, total_bits - 2);
                while (k2 == k1) k2 = rand_int(1, total_bits - 2);
                if (k1 > k2) std::swap(k1, k2);
            }

            for (int i = 0; i < total_bits; ++i) {
                nb1[i] = (i >= k1 && i < k2) ? bits_b[i] : bits_a[i];
                nb2[i] = (i >= k1 && i < k2) ? bits_a[i] : bits_b[i];
            }
        }

        auto bits_to_ind = [&](const std::vector<uint8_t>& bits) {
            std::vector<uint64_t> gc(a.genes.size(), 0);
            int pos = 0;
            for (int i = 0; i < (int)a.genes.size(); ++i) {
                uint64_t v = 0;
                for (int bit = a.genes[i] - 1; bit >= 0; --bit)
                    v |= ((uint64_t)bits[pos++] << bit);
                gc[i] = v;
            }
            return std::make_shared<GrayIndividual>(gc, a.x_min, a.x_max, a.genes, func);
        };

        return {bits_to_ind(nb1), bits_to_ind(nb2)};
    }

private:
    void _init_steps() {
        int dim = (int)genes.size();
        steps.resize(dim);
        for (int i = 0; i < dim; ++i)
            steps[i] = (x_max[i] - x_min[i]) / (double)((1ULL << genes[i]) - 1);
    }
};

struct RealIndividual : Individual {
    std::vector<double> x;

    RealIndividual(
        std::vector<double> x_,
        const std::vector<double>& xmin,
        const std::vector<double>& xmax,
        const FuncT& func)
        : Individual(xmin, xmax, IndividualType::Real), x(std::move(x_))
    {
        f_value = func(x);
    }

    RealIndividual(
        std::vector<double> x_,
        const std::vector<double>& xmin,
        const std::vector<double>& xmax,
        double f_val)
        : Individual(xmin, xmax, IndividualType::Real), x(std::move(x_))
    {
        f_value = f_val;
    }

    std::vector<double> real_x() const override { return x; }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<RealIndividual>(x, x_min, x_max, f_value);
        p->ep_n = ep_n;
        return p;
    }

    void update_f(const FuncT& func) { f_value = func(x); }

    void mutation(const FuncT& func, double pmax) {
        ep_n += 1;
        int dim = (int)x.size();
        for (int i = 0; i < dim; ++i) {
            double perturb = pmax * rand_uniform(-std::abs(x[i]), std::abs(x[i])) / ep_n;
            x[i] = std::clamp(x[i] + perturb, x_min[i], x_max[i]);
        }
        update_f(func);
    }

    // void mutation(const FuncT& func, double pmax) {
    //     ep_n += 1;
    //     int dim = (int)x.size();
    //     for (int i = 0; i < dim; ++i) {
    //         double range = x_max[i] - x_min[i];
    //         double perturb = pmax * rand_uniform(-0.5, 0.5) * range / ep_n;
    //         x[i] = std::clamp(x[i] + perturb, x_min[i], x_max[i]);
    //     }
    //     update_f(func);
    // }

//     static std::shared_ptr<RealIndividual> crossover(
//         const RealIndividual& a, const RealIndividual& b,
//         double alpha,
//         const std::vector<double>& xmin,
//         const std::vector<double>& xmax,
//         const FuncT& func)
//     {
//         int dim = (int)a.x.size();
//         std::vector<double> new_x(dim);
//         for (int i = 0; i < dim; ++i) {
//             double lo = std::min(a.x[i], b.x[i]);
//             double hi = std::max(a.x[i], b.x[i]);
//             double I  = hi - lo;
//             new_x[i]  = std::clamp(rand_uniform(lo - I * alpha, hi + I * alpha),
//                                    xmin[i], xmax[i]);
//         }
//         return std::make_shared<RealIndividual>(std::move(new_x), xmin, xmax, func);
//     }
// };
    static std::shared_ptr<RealIndividual> crossover(
    const RealIndividual& a,
    const RealIndividual& b,
    double alpha,
    const std::vector<double>& xmin,
    const std::vector<double>& xmax,
    const FuncT& func) {
        const Eigen::Index dim = static_cast<Eigen::Index>(a.x.size());

        Eigen::Map<const Eigen::ArrayXd> ax(a.x.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> bx(b.x.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> min_x(xmin.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> max_x(xmax.data(), dim);

        const Eigen::ArrayXd lo = ax.min(bx);
        const Eigen::ArrayXd hi = ax.max(bx);
        const Eigen::ArrayXd interval = hi - lo;

        const Eigen::ArrayXd lower = lo - alpha * interval;
        const Eigen::ArrayXd upper = hi + alpha * interval;

        Eigen::ArrayXd u(dim);
        for (Eigen::Index i = 0; i < dim; ++i) {
            u[i] = rand_uniform(0.0, 1.0);
        }

        Eigen::ArrayXd child = lower + u * (upper - lower);

        // Эквивалент std::clamp(value, xmin[i], xmax[i]) для всего вектора.
        child = child.max(min_x).min(max_x);

        std::vector<double> new_x(
            child.data(),
            child.data() + child.size()
        );

        return std::make_shared<RealIndividual>(
            std::move(new_x), xmin, xmax, func
        );
    }
    };

struct Country {
    std::vector<std::shared_ptr<Individual>> population;
    std::vector<double> x_min, x_max;
    std::vector<int>    genes;
    FuncT               f;
    int                 N;
    IndividualType      itype;

    int      action = -1;
    Country* ally   = nullptr;
    Country* enemy  = nullptr;

    Country(int N_,
            const std::vector<double>& xmin,
            const std::vector<double>& xmax,
            const FuncT& func,
            IndividualType it,
            const std::vector<int>& genes_ = {})
        : x_min(xmin), x_max(xmax), genes(genes_), f(func), N(N_), itype(it)
    {
        int dim = (int)xmin.size();
        population.reserve(N);

        if (itype == IndividualType::Gray) {
            std::vector<uint64_t> lmin(dim), lmax(dim);
            for (int d = 0; d < dim; ++d) {
                uint64_t maxval = (1ULL << genes[d]) - 1;
                lmin[d] = rand_uint64(0, maxval - 1);
                lmax[d] = rand_uint64(lmin[d] + 1, maxval);
            }
            for (int i = 0; i < N; ++i) {
                std::vector<uint64_t> dec(dim);
                for (int d = 0; d < dim; ++d)
                    dec[d] = rand_uint64(lmin[d], lmax[d]);
                population.push_back(GrayIndividual::from_decimal(dec, xmin, xmax, genes, f));
            }
        } else {
            std::vector<double> lmin(dim), lmax(dim);
            for (int d = 0; d < dim; ++d) {
                lmin[d] = rand_uniform(xmin[d], xmax[d]);
                lmax[d] = rand_uniform(lmin[d], xmax[d]);
            }
            for (int i = 0; i < N; ++i) {
                std::vector<double> x(dim);
                for (int d = 0; d < dim; ++d)
                    x[d] = rand_uniform(lmin[d], lmax[d]);
                population.push_back(std::make_shared<RealIndividual>(x, xmin, xmax, f));
            }
        }
        sort_population();
    }

    bool empty() const noexcept { return population.empty(); }
    int  size()  const noexcept { return (int)population.size(); }

    void sort_population() {
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a->f_value < b->f_value; });
    }

    double best_f() const noexcept {
        return population.empty() ? std::numeric_limits<double>::infinity()
                                  : population[0]->f_value;
    }

    double avg_f() const noexcept {
        if (population.empty()) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (const auto& ind : population) s += ind->f_value;
        return s / (double)population.size();
    }

    void select_action(std::vector<Country*>& all_countries,
                        double p_war, double p_trade,
                        double p_motion, double p_epidemic) {
        action = weighted_action_choice(p_war, p_trade, p_motion, p_epidemic);
        if (action == 1) {
            std::vector<Country*> candidates;
            for (auto* c : all_countries)
                if (c->action == -1 && c != this) candidates.push_back(c);
            if (!candidates.empty()) {
                auto* chosen = candidates[rand_int(0, (int)candidates.size() - 1)];
                ally = chosen;
                chosen->action = 1;
                chosen->ally   = this;
            } else {
                action = (rand_int(0, 1) == 0) ? 0 : 3;
            }
        } else if (action == 2) {
            std::vector<Country*> candidates;
            for (auto* c : all_countries)
                if (c->action == -1 && c != this) candidates.push_back(c);
            if (!candidates.empty()) {
                auto* chosen = candidates[rand_int(0, (int)candidates.size() - 1)];
                enemy = chosen;
                chosen->action = 2;
                chosen->enemy  = this;
            } else {
                action = (rand_int(0, 1) == 0) ? 0 : 3;
            }
        }
    }

void do_motion() {
    if (population.empty()) { action = -1; return; }

    if (itype == IndividualType::Gray) {
        auto best_dec = std::static_pointer_cast<GrayIndividual>(population[0])->decimal_x();
        for (int i = 1; i < size(); ++i) {
            auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
            auto dec = ind->decimal_x();
            double r = rand_uniform(0.0, 2.0);
            std::vector<uint64_t> new_dec(genes.size());
            for (int d = 0; d < (int)genes.size(); ++d) {
                int64_t diff = (int64_t)best_dec[d] - (int64_t)dec[d];
                int64_t nd   = (int64_t)dec[d] + (int64_t)(r * (double)diff);
                uint64_t maxval = (1ULL << genes[d]) - 1;
                new_dec[d] = (uint64_t)std::clamp(nd, (int64_t)0, (int64_t)maxval);
            }
            population[i] = GrayIndividual::from_decimal(new_dec, x_min, x_max, genes, f);
        }
    } else {
        const auto& best_x = std::static_pointer_cast<RealIndividual>(population[0])->x;
        for (int i = 1; i < size(); ++i) {
            auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
            int dim  = (int)ind->x.size();
            double r = rand_uniform(0.0, 2.0);   // один коэффициент на особь
            for (int d = 0; d < dim; ++d) {
                ind->x[d] = std::clamp(ind->x[d] + r * (best_x[d] - ind->x[d]),
                                       x_min[d], x_max[d]);
            }
            ind->update_f(f);
        }
    }
    sort_population();
    action = -1;
}
    // void do_motion(double r_max) {
    //     if (population.empty()) {
    //         action = -1;
    //         return;
    //     }
    //
    //     if (itype == IndividualType::Gray) {
    //         auto best_dec =
    //             std::static_pointer_cast<GrayIndividual>(population[0])->decimal_x();
    //
    //         for (int i = 1; i < size(); ++i) {
    //             auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
    //             auto dec = ind->decimal_x();
    //
    //             double r = rand_uniform(0.0, r_max);
    //             std::vector<uint64_t> new_dec(genes.size());
    //
    //             for (int d = 0; d < (int)genes.size(); ++d) {
    //                 int64_t diff =
    //                     static_cast<int64_t>(best_dec[d]) -
    //                     static_cast<int64_t>(dec[d]);
    //
    //                 int64_t nd =
    //                     static_cast<int64_t>(dec[d]) +
    //                     static_cast<int64_t>(r * static_cast<double>(diff));
    //
    //                 uint64_t maxval = (1ULL << genes[d]) - 1;
    //
    //                 new_dec[d] = static_cast<uint64_t>(
    //                     std::clamp(nd, int64_t{0}, static_cast<int64_t>(maxval))
    //                 );
    //             }
    //
    //             population[i] =
    //                 GrayIndividual::from_decimal(new_dec, x_min, x_max, genes, f);
    //         }
    //     } else {
    //         const auto& best_x =
    //             std::static_pointer_cast<RealIndividual>(population[0])->x;
    //
    //         for (int i = 1; i < size(); ++i) {
    //             auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
    //
    //             // Один r на особь: движение остаётся вдоль направления
    //             // «текущая особь -> лидер», что важно для повёрнутой F1.
    //             double r = rand_uniform(0.0, r_max);
    //
    //             for (int d = 0; d < (int)ind->x.size(); ++d) {
    //                 ind->x[d] = std::clamp(
    //                     ind->x[d] + r * (best_x[d] - ind->x[d]),
    //                     x_min[d],
    //                     x_max[d]
    //                 );
    //             }
    //
    //             ind->update_f(f);
    //         }
    //     }
    //
    //     sort_population();
    //     action = -1;
    // }

    void do_epidemic(double elite_frac, double dead_frac, double pmax_real) {
        double q_max_term_gray = (itype == IndividualType::Gray) ? pmax_real : 0.0;
        int n       = size();
        int n_elite = (int)std::ceil(elite_frac * n);
        int n_dead  = (int)std::ceil(dead_frac  * n);

        if (n_dead >= n) { population.clear(); action = -1; return; }
        if (n_dead > 0)
            population.erase(population.end() - n_dead, population.end());

        for (int i = n_elite; i < size(); ++i) {
            if (itype == IndividualType::Gray) {
                auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
                auto new_ind = ind->mutate(q_max_term_gray, f);
                new_ind->ep_n = ind->ep_n + 1;
                population[i] = new_ind;
            } else {
                auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
                ind->mutation(f, pmax_real);     // ep_n там уже растет внутри mutation()
            }
        }
        sort_population();
        action = -1;
    }

    void reproduction(int n_min, int n_max,
                      double p_min, double p_max,
                      double f_min, double f_max,
                      int ti, int t_max) {
        if (size() < 2) return;
        double avg   = avg_f();
        double n_frac = (f_max - avg) / (f_max - f_min);
        int n = std::clamp((int)std::ceil((n_max - n_min) * n_frac + n_min), n_min, n_max);

        if (itype == IndividualType::Gray) {
            for (int i = 0; i < n; ++i) {
                int k1 = rand_int(0, size() - 1);
                int k2 = k1;
                while (k2 == k1) k2 = rand_int(0, size() - 1);
                auto& a = *std::static_pointer_cast<GrayIndividual>(population[k1]);
                auto& b = *std::static_pointer_cast<GrayIndividual>(population[k2]);
                auto [c1, c2] = GrayIndividual::crossover(a, b, f);
                population.push_back(std::move(c1));
                population.push_back(std::move(c2));
            }
        } else {
            double p = std::clamp(
                (p_max - p_min) * (1.0 - (double)ti / t_max) *
                (avg - f_min) / (f_max - f_min) + p_min,
                p_min, p_max);
            for (int i = 0; i < 2 * n; ++i) {
                int k1 = rand_int(0, size() - 1);
                int k2 = k1;
                while (k2 == k1) k2 = rand_int(0, size() - 1);
                auto& a = *std::static_pointer_cast<RealIndividual>(population[k1]);
                auto& b = *std::static_pointer_cast<RealIndividual>(population[k2]);
                population.push_back(RealIndividual::crossover(a, b, p, x_min, x_max, f));
            }
        }
        sort_population();
    }

    void extinction(int m_min, int m_max, double f_min, double f_max) {
        double avg = avg_f();
        int m = std::clamp(
            (int)((m_max - m_min) * (avg - f_min) / (f_max - f_min) + m_min),
            m_min, m_max);
        if (m >= size()) { population.clear(); return; }
        population.erase(population.end() - m, population.end());
    }

    void truncate(int max_size) {
        if (size() > max_size) population.resize(max_size);
    }

    void update_individual_type() {
        int dim = (int)x_min.size();
        for (auto& ind : population) {
            if (ind->itype == itype) continue;
            if (itype == IndividualType::Real) {
                auto rx   = ind->real_x();
                auto ni   = std::make_shared<RealIndividual>(rx, x_min, x_max, ind->f_value);
                ni->ep_n  = ind->ep_n;
                ind       = ni;
            } else {
                auto rx = ind->real_x();
                std::vector<uint64_t> gc(dim);
                for (int d = 0; d < dim; ++d) {
                    double step    = (x_max[d] - x_min[d]) / (double)((1ULL << genes[d]) - 1);
                    int64_t v      = (int64_t)std::round((rx[d] - x_min[d]) / step);
                    uint64_t maxv  = (1ULL << genes[d]) - 1;
                    gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)maxv));
                }
                auto ni = std::make_shared<GrayIndividual>(gc, x_min, x_max, genes, ind->f_value);
                ni->ep_n = ind->ep_n;
                ind = ni;
            }
        }
    }

    std::shared_ptr<Individual> make_random_individual() const {
        const int dim = static_cast<int>(x_min.size());

        if (itype == IndividualType::Gray) {
            std::vector<uint64_t> decimal(dim);

            for (int d = 0; d < dim; ++d) {
                const uint64_t maxval = (1ULL << genes[d]) - 1;
                decimal[d] = rand_uint64(0, maxval);
            }

            return GrayIndividual::from_decimal(decimal, x_min, x_max, genes, f);
        }

        std::vector<double> x(dim);
        for (int d = 0; d < dim; ++d) {
            x[d] = rand_uniform(x_min[d], x_max[d]);
        }

        return std::make_shared<RealIndividual>(std::move(x), x_min, x_max, f);
    }

    static void do_trade(Country& c1, Country& c2, int k) {
        int actual_k = k;
        if (c1.size() <= k || c2.size() <= k)
            actual_k = std::min(c1.size(), c2.size()) / 2;
        if (actual_k == 0) {
            c1.action = -1; c2.action = -1; c1.ally = nullptr; c2.ally = nullptr;
            return;
        }
        auto pick_indices = [](int sz, int cnt) {
            std::vector<int> idx(sz);
            std::iota(idx.begin(), idx.end(), 0);
            for (int i = 0; i < cnt; ++i) {
                int j = rand_int(i, sz - 1);
                std::swap(idx[i], idx[j]);
            }
            idx.resize(cnt);
            return idx;
        };
        auto remove_by_idx = [](std::vector<std::shared_ptr<Individual>>& pop,
                                std::vector<int> idx) {
            std::sort(idx.begin(), idx.end(), std::greater<int>());
            for (int i : idx) pop.erase(pop.begin() + i);
        };

        auto idx1 = pick_indices(c1.size(), actual_k);
        auto idx2 = pick_indices(c2.size(), actual_k);

        std::vector<std::shared_ptr<Individual>> t1, t2;
        for (int i : idx1) t1.push_back(c1.population[i]->clone());
        for (int i : idx2) t2.push_back(c2.population[i]->clone());

        remove_by_idx(c1.population, idx1);
        remove_by_idx(c2.population, idx2);
        for (auto& t : t2) c1.population.push_back(t);
        for (auto& t : t1) c2.population.push_back(t);

        c1.update_individual_type(); c2.update_individual_type();
        c1.sort_population();        c2.sort_population();
        c1.action = -1; c2.action = -1; c1.ally = nullptr; c2.ally = nullptr;
    }

    static std::shared_ptr<Individual> recruit_from_duel(
    const Individual& winner,
    const Individual& loser,
    Country& home,
    double r_max)
{
    auto wx = winner.real_x();
    auto lx = loser.real_x();
    const int dim = static_cast<int>(wx.size());

    double dist2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        double diff = wx[d] - lx[d];
        dist2 += diff * diff;
    }
    const double dist = std::sqrt(dist2);

    // Случайное направление шума, одно на всю особь.
    std::vector<double> noise(dim);
    double nrm2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        noise[d] = rand_uniform(-1.0, 1.0);
        nrm2 += noise[d] * noise[d];
    }
    const double nrm = std::sqrt(nrm2) + 1e-12;

    // alpha: насколько далеко отражаем за победителя.
    // jitter: слабый изотропный сдвиг, чтобы не ходить только по одной прямой.
    const double alpha = rand_uniform(0.1, std::max(0.15, r_max * 0.4));
    const double jitter = 0.05 * dist;

    std::vector<double> nx(dim);
    for (int d = 0; d < dim; ++d) {
        const double reflected = wx[d] + alpha * (wx[d] - lx[d]);
        nx[d] = std::clamp(
            reflected + jitter * noise[d] / nrm,
            home.x_min[d],
            home.x_max[d]
        );
    }

    if (home.itype == IndividualType::Real) {
        return std::make_shared<RealIndividual>(
            std::move(nx), home.x_min, home.x_max, home.f
        );
    }

    std::vector<uint64_t> dec(dim);
    for (int d = 0; d < dim; ++d) {
        const double step =
            (home.x_max[d] - home.x_min[d]) /
            static_cast<double>((1ULL << home.genes[d]) - 1);
        const int64_t v = static_cast<int64_t>(
            std::round((nx[d] - home.x_min[d]) / step)
        );
        const uint64_t maxv = (1ULL << home.genes[d]) - 1;
        dec[d] = static_cast<uint64_t>(
            std::clamp(v, int64_t{0}, static_cast<int64_t>(maxv))
        );
    }
    return GrayIndividual::from_decimal(dec, home.x_min, home.x_max, home.genes, home.f);
    }

    static void do_war(Country& c1, Country& c2, int l) {
    int actual_l = l;

    if (c1.size() <= l || c2.size() <= l) {
        actual_l = std::min(c1.size(), c2.size());
    }

    if (actual_l == 0) {
        c1.action = -1;
        c2.action = -1;
        c1.enemy = nullptr;
        c2.enemy = nullptr;
        return;
    }

    auto pick_indices = [](int sz, int cnt) {
        std::vector<int> idx(sz);
        std::iota(idx.begin(), idx.end(), 0);

        for (int i = 0; i < cnt; ++i) {
            int j = rand_int(i, sz - 1);
            std::swap(idx[i], idx[j]);
        }

        idx.resize(cnt);
        return idx;
    };

    auto remove_by_idx = [](std::vector<std::shared_ptr<Individual>>& pop,
                            std::vector<int> idx) {
        std::sort(idx.begin(), idx.end(), std::greater<int>());

        for (int i : idx) {
            pop.erase(pop.begin() + i);
        }
    };

    const auto idx1 = pick_indices(c1.size(), actual_l);
    const auto idx2 = pick_indices(c2.size(), actual_l);

    std::vector<std::shared_ptr<Individual>> war1;
    std::vector<std::shared_ptr<Individual>> war2;
    war1.reserve(actual_l);
    war2.reserve(actual_l);

    for (int i : idx1) {
        war1.push_back(c1.population[i]->clone());
    }

    for (int i : idx2) {
        war2.push_back(c2.population[i]->clone());
    }

    // Призванные на войну временно удаляются из стран.
    remove_by_idx(c1.population, idx1);
    remove_by_idx(c2.population, idx2);

    int wins1 = 0;
    int wins2 = 0;

    // Только эти воины могут стать пленными после итоговой победы страны.
    std::vector<std::shared_ptr<Individual>> survivors1;
    std::vector<std::shared_ptr<Individual>> survivors2;
    survivors1.reserve(actual_l);
    survivors2.reserve(actual_l);

    for (int i = 0; i < actual_l; ++i) {
        if (*war1[i] < *war2[i]) {
            // c1 выигрывает дуэль: её воин выживает.
            ++wins1;
            survivors1.push_back(war1[i]);

            // Проигравший c2 заменяется новой случайной особью в c2.
            c2.population.push_back(c2.make_random_individual());

        } else if (*war2[i] < *war1[i]) {
            // c2 выигрывает дуэль: её воин выживает.
            ++wins2;
            survivors2.push_back(war2[i]);

            // Проигравший c1 заменяется новой случайной особью в c1.
            c1.population.push_back(c1.make_random_individual());

        } else {
            // Ничья в конкретной дуэли: оба воина остаются живы.
            survivors1.push_back(war1[i]);
            survivors2.push_back(war2[i]);
        }
    }

    if (wins1 > wins2) {
        // c1 победила в войне.
        // Все выжившие, включая выживших c2, переходят в c1.
        for (auto& warrior : survivors1) {
            c1.population.push_back(warrior);
        }

        for (auto& prisoner : survivors2) {
            c1.population.push_back(prisoner);
        }

    } else if (wins2 > wins1) {
        // c2 победила в войне.
        // Все выжившие, включая выживших c1, переходят в c2.
        for (auto& prisoner : survivors1) {
            c2.population.push_back(prisoner);
        }

        for (auto& warrior : survivors2) {
            c2.population.push_back(warrior);
        }

    } else {
        // Ничья по числу выигранных дуэлей:
        // выжившие возвращаются в исходные страны.
        for (auto& warrior : survivors1) {
            c1.population.push_back(warrior);
        }

        for (auto& warrior : survivors2) {
            c2.population.push_back(warrior);
        }
    }

    // Пленные приводятся к представлению страны-победителя.
    c1.update_individual_type();
    c2.update_individual_type();

    c1.sort_population();
    c2.sort_population();

    c1.action = -1;
    c2.action = -1;
    c1.enemy = nullptr;
    c2.enemy = nullptr;
}
//     static void do_war(Country& c1, Country& c2, int l, double r_max) {
//     int actual_l = l;
//     if (c1.size() <= l || c2.size() <= l)
//         actual_l = std::min(c1.size(), c2.size());
//
//     if (actual_l == 0) {
//         c1.action = -1; c2.action = -1;
//         c1.enemy = nullptr; c2.enemy = nullptr;
//         return;
//     }
//
//     auto pick_indices = [](int sz, int cnt) {
//         std::vector<int> idx(sz);
//         std::iota(idx.begin(), idx.end(), 0);
//         for (int i = 0; i < cnt; ++i)
//             std::swap(idx[i], idx[rand_int(i, sz - 1)]);
//         idx.resize(cnt);
//         return idx;
//     };
//     auto remove_by_idx = [](auto& pop, std::vector<int> idx) {
//         std::sort(idx.begin(), idx.end(), std::greater<int>());
//         for (int i : idx) pop.erase(pop.begin() + i);
//     };
//
//     const auto idx1 = pick_indices(c1.size(), actual_l);
//     const auto idx2 = pick_indices(c2.size(), actual_l);
//
//     std::vector<std::shared_ptr<Individual>> war1, war2;
//     for (int i : idx1) war1.push_back(c1.population[i]->clone());
//     for (int i : idx2) war2.push_back(c2.population[i]->clone());
//
//     remove_by_idx(c1.population, idx1);
//     remove_by_idx(c2.population, idx2);
//
//     int wins1 = 0, wins2 = 0;
//     std::vector<std::shared_ptr<Individual>> survivors1, survivors2;
//
//     for (int i = 0; i < actual_l; ++i) {
//         if (*war1[i] < *war2[i]) {
//             ++wins1;
//             survivors1.push_back(war1[i]);
//             c2.population.push_back(
//                 recruit_from_duel(*war1[i], *war2[i], c2, r_max)
//             );
//         } else if (*war2[i] < *war1[i]) {
//             ++wins2;
//             survivors2.push_back(war2[i]);
//             c1.population.push_back(
//                 recruit_from_duel(*war2[i], *war1[i], c1, r_max)
//             );
//         } else {
//             survivors1.push_back(war1[i]);
//             survivors2.push_back(war2[i]);
//         }
//     }
//
//     auto assimilate = [&](Country& winner, std::shared_ptr<Individual> prisoner) {
//         if (winner.empty()) {
//             winner.population.push_back(prisoner);
//             return;
//         }
//         const auto capital = winner.population[0]->real_x();
//         auto px = prisoner->real_x();
//         const double r = rand_uniform(0.35, 0.85);
//         for (int d = 0; d < (int)px.size(); ++d) {
//             px[d] = std::clamp(
//                 px[d] + r * (capital[d] - px[d]),
//                 winner.x_min[d], winner.x_max[d]
//             );
//         }
//         if (winner.itype == IndividualType::Real) {
//             auto ni = std::make_shared<RealIndividual>(
//                 std::move(px), winner.x_min, winner.x_max, winner.f
//             );
//             ni->ep_n = prisoner->ep_n;
//             winner.population.push_back(std::move(ni));
//         } else {
//             // проще: сначала положить как есть, тип поправит update_individual_type
//             winner.population.push_back(prisoner);
//         }
//     };
//
//     if (wins1 > wins2) {
//         for (auto& w : survivors1) c1.population.push_back(w);
//         for (auto& p : survivors2) assimilate(c1, p);
//     } else if (wins2 > wins1) {
//         for (auto& p : survivors1) assimilate(c2, p);
//         for (auto& w : survivors2) c2.population.push_back(w);
//     } else {
//         for (auto& w : survivors1) c1.population.push_back(w);
//         for (auto& w : survivors2) c2.population.push_back(w);
//     }
//
//     c1.update_individual_type();
//     c2.update_individual_type();
//     c1.sort_population();
//     c2.sort_population();
//     c1.action = -1; c2.action = -1;
//     c1.enemy = nullptr; c2.enemy = nullptr;
// }
};

class CountriesAlgorithm {
public:
    using Vec = std::vector<double>;

    struct Params {
        std::vector<double> x_min, x_max;
        std::vector<int>    genes;
        double p_min        = 0.1;
        double p_max        = 0.5;
        int    M            = 10;
        int    N            = 20;
        int    n_min        = 1;
        int    n_max        = 5;
        int    m_min        = 1;
        int    m_max        = 3;
        int    k            = 3;
        int    l            = 3;
        double ep_elite     = 0.2;
        double ep_dead      = 0.3;
        int    max_mutation = 3;
        int    tmax         = 1000;
        double gray_percent = 0.5;
        bool   printing     = true;

        // Вероятности выбора действий стран на каждой итерации.
        // Порядок: Война, Обмен, Движение к лидеру, Эпидемия.
        // Сумма всех четырёх значений должна быть равна 1.0.
        double p_war      = 0.25;
        double p_trade    = 0.25;
        double p_motion   = 0.25;
        double p_epidemic = 0.25;
        bool   adaptive_actions = true;
        double action_alpha     = 0.1;   // скорость обучения EMA
        double action_pmin      = 0.07;
        double action_warmup_frac = 0.06;
    };

    // Новый компонент — держим credit/вероятности вне Country, на уровне алгоритма.
    struct ActionAdaptation {
        // Порядок: 0=motion, 1=trade, 2=war, 3=epidemic
        std::array<double, 4> reward = {0.0, 0.0, 0.0, 0.0};   // EMA "выгоды на вызов ЦФ"
        std::array<double, 4> probs  = {0.25, 0.25, 0.25, 0.25};

        // credit = относительное улучшение avg_f страны / (число вызовов ЦФ, потраченных действием)
        void update(int action_idx, double f_before, double f_after, long calls_spent, double alpha) {
            if (calls_spent <= 0) calls_spent = 1;
            double improvement = std::max(0.0, f_before - f_after); // минимизация
            double credit = improvement / (double)calls_spent;
            reward[action_idx] = (1.0 - alpha) * reward[action_idx] + alpha * credit;
        }

        // Probability matching с нижним порогом (аналог adaptive pursuit)
        void renormalize(double p_min) {
            double sum = 0.0;
            for (double r : reward) sum += r;
            if (sum <= 1e-15) { probs = {0.25, 0.25, 0.25, 0.25}; return; }
            double residual = 1.0 - 4 * p_min;
            for (int i = 0; i < 4; ++i)
                probs[i] = p_min + residual * (reward[i] / sum);
        }
    };

    explicit CountriesAlgorithm(FuncT func, Params params)
        : p_(std::move(params)), calls_count_(std::make_shared<long>(0))
    {
        {
            double sum_p = p_.p_war + p_.p_trade + p_.p_motion + p_.p_epidemic;
            if (std::abs(sum_p - 1.0) > 1e-9) {
                throw std::invalid_argument(
                    "Params: p_war + p_trade + p_motion + p_epidemic must sum to 1.0 (got "
                    + std::to_string(sum_p) + ")");
            }
        }
        // Оборачиваем функцию для подсчета вызовов.
        // Используем shared_ptr, чтобы счетчик не терялся при копировании/перемещении
        // лямбды или класса CountriesAlgorithm внутри std::function.
        auto counter = calls_count_;
        f_ = [counter, func](const std::vector<double>& x) {
            (*counter)++;
            return func(x);
        };

        int gray_M = (int)(p_.gray_percent * p_.M);
        int real_M = p_.M - gray_M;
        countries_.reserve(p_.M);
        for (int i = 0; i < gray_M; ++i)
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.x_min, p_.x_max, f_, IndividualType::Gray, p_.genes));
        for (int i = 0; i < real_M; ++i)
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.x_min, p_.x_max, f_, IndividualType::Real, p_.genes));
    }

    std::tuple<Vec, double, long> start(const Vec& canonical_x,
                                    double epsilon,
                                    std::optional<double> y_epsilon,
                                    std::optional<long> max_calls) {

    double canonical_y = f_(canonical_x);
    Vec best_x;
    double best_f = std::numeric_limits<double>::infinity();
    long ti = 0;
    if (!countries_.empty() && !countries_[0]->population.empty()) {
        best_x = countries_[0]->population[0]->real_x();
        best_f = countries_[0]->population[0]->f_value;
    }

    // Разогрев: первые action_warmup_frac * tmax итераций вероятности
    // действий не адаптируются, чтобы EMA-статистика набралась не на шуме.
    long warmup_iters = static_cast<long>(
        std::round(p_.action_warmup_frac * static_cast<double>(p_.tmax)));

    for (ti = 1; ti <= p_.tmax; ++ti) {
        double progress = static_cast<double>(ti - 1) /
      std::max(1, p_.tmax - 1);
        // std::cout << p_.p_war << ' ' << p_.p_trade << ' ' << p_.p_motion << ' ' << p_.p_epidemic << std::endl;
        // От 2.0 в начале до 1.2 на последней итерации.
        // Степень 0.6 означает медленное уменьшение в начале.
        double r_max = 2.0 - 0.8 * std::pow(progress, 0.6);
        // Проверка лимита вызовов перед итерацией
        if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
            if (p_.printing) std::cout << "Лимит вызовов функции достигнут\n";
            return {best_x, best_f, ti};
        }

        {
            std::vector<Country*> ptrs;
            ptrs.reserve(countries_.size());
            for (auto& c : countries_) ptrs.push_back(c.get());
            for (auto* c : ptrs)
                if (c->action == -1)
                    c->select_action(ptrs, p_.p_war, p_.p_trade, p_.p_motion, p_.p_epidemic);
        }

        int cnt_motion=0, cnt_trade=0, cnt_war=0, cnt_epi=0;
        double q_max_term = (1.0 - (double)ti / p_.tmax) * p_.max_mutation;

        // "avg_f до" по каждой стране — нужно для расчёта credit действия.
        std::vector<double> avg_before(countries_.size());
        for (size_t i = 0; i < countries_.size(); ++i)
            avg_before[i] = countries_[i]->avg_f();

        for (size_t i = 0; i < countries_.size(); ++i) {
            auto& c = countries_[i];
            int act = c->action;
            long calls_before = *calls_count_;

            if (act == 0) {
                ++cnt_motion;
                c->do_motion();
            } else if (act == 1 && c->ally != nullptr) {
                ++cnt_trade;
                Country::do_trade(*c, *c->ally, p_.k);
            } else if (act == 2 && c->enemy != nullptr) {
                ++cnt_war;
                Country::do_war(*c, *c->enemy, p_.l);
            } else if (act == 3) {
                ++cnt_epi;
                double pmax = (c->itype == IndividualType::Gray)
                                ? q_max_term      // это (1 - t/tmax) * q_max, без -n_ep и без max(1,...)
                                : p_.p_max;
                c->do_epidemic(p_.ep_elite, p_.ep_dead, pmax);
            }

            // Обновляем credit только если это действие реально выполнялось
            // (act != -1). Для Trade/War партнёр к этому моменту уже имеет
            // action == -1 (сброшен внутри do_trade/do_war), поэтому здесь
            // естественным образом кредит приписывается только инициатору.
            if (p_.adaptive_actions && act != -1) {
                long calls_after = *calls_count_;
                double f_after = c->avg_f();
                action_adapt_.update(act, avg_before[i], f_after,
                                      calls_after - calls_before, p_.action_alpha);
            }
        }

        // Пересчёт вероятностей действий на следующую итерацию (после разогрева).
        if (p_.adaptive_actions && ti > warmup_iters) {
            action_adapt_.renormalize(p_.action_pmin);
            p_.p_motion   = action_adapt_.probs[0];
            p_.p_trade    = action_adapt_.probs[1];
            p_.p_war      = action_adapt_.probs[2];
            p_.p_epidemic = action_adapt_.probs[3];
        }

        _remove_empty();
        if (countries_.empty()) break;

        std::sort(countries_.begin(), countries_.end(),
                  [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });

        double f_min = countries_.front()->avg_f();
        double f_max = countries_.back()->avg_f();

        if (f_min == f_max) {
            std::sort(countries_.begin(), countries_.end(),
                      [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });
            best_x = countries_[0]->population[0]->real_x();
            best_f = countries_[0]->population[0]->f_value;
            if (p_.printing) std::cout << "fmin == fmax\n";
            return {best_x, best_f, ti};
        }

        std::vector<std::shared_ptr<Individual>> e_individuals;
        for (auto& c : countries_) {
            if (c->size() == 1) {
                e_individuals.push_back(c->population[0]);
                continue;
            }
            c->reproduction(p_.n_min, p_.n_max, p_.p_min, p_.p_max,
                            f_min, f_max, ti, p_.tmax);
            c->extinction(p_.m_min, p_.m_max, f_min, f_max);
        }

        _remove_empty();

        if (!countries_.empty()) {
            for (auto& ind : e_individuals)
                _add_individual_to_random_country(ind);
        }

        for (auto& c : countries_) c->truncate(2 * p_.N);

        std::sort(countries_.begin(), countries_.end(),
                  [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

        if (countries_.empty()) break;

        best_x = countries_[0]->population[0]->real_x();
        best_f = countries_[0]->population[0]->f_value;

        if (p_.printing) {
            std::cout << ti << ") f=" << best_f << std::endl;
        }

        // 1) Остановка по Евклидову расстоянию x до оптимума (epsilon)
        double dist = 0.0;
        for (size_t i = 0; i < best_x.size(); ++i) {
            dist += (best_x[i] - canonical_x[i]) * (best_x[i] - canonical_x[i]);
        }
        dist = std::sqrt(dist);

        // if (dist <= epsilon) {
        //     return {best_x, best_f, ti};
        // }

        // 2) Остановка по близости f(x) к f(canonical_x) (y_epsilon)
        // if (y_epsilon.has_value() && std::abs(best_f - canonical_y) <= y_epsilon.value()) {
        //     return {best_x, best_f, ti};
        // }
        //
        // if (best_f < canonical_y) {
        //     return {best_x, best_f, ti};
        // }

        // 3) Остановка по лимиту вызовов внутри итерации
        if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
            if (p_.printing) std::cout << "Лимит вызовов функции достигнут\n";
            return {best_x, best_f, ti};
        }
    }

    return {best_x, best_f, ti > p_.tmax ? p_.tmax : ti};
}

    // Аналог testing(canonical_x, epsilon, y_epsilon, max_calls) -> (best_x, best_f, ti)
    // std::tuple<Vec, double, long> start(const Vec& canonical_x,
    //                                     double epsilon,
    //                                     std::optional<double> y_epsilon,
    //                                     std::optional<long> max_calls) {
    //
    //     double canonical_y = f_(canonical_x);
    //     Vec best_x;
    //     double best_f = std::numeric_limits<double>::infinity();
    //     long ti = 0;
    //     if (!countries_.empty() && !countries_[0]->population.empty()) {
    //         best_x = countries_[0]->population[0]->real_x();
    //         best_f = countries_[0]->population[0]->f_value;
    //     }
    //
    //     for (ti = 1; ti <= p_.tmax; ++ti) {
    //         double progress = static_cast<double>(ti - 1) /
    //       std::max(1, p_.tmax - 1);
    //
    //         // От 2.0 в начале до 1.2 на последней итерации.
    //         // Степень 0.6 означает медленное уменьшение в начале.
    //         // double r_max = 2.0 - 0.8 * std::pow(progress, 0.6);
    //         // Проверка лимита вызовов перед итерацией
    //         if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
    //             if (p_.printing) std::cout << "Лимит вызовов функции достигнут\n";
    //             return {best_x, best_f, ti};
    //         }
    //
    //         {
    //             std::vector<Country*> ptrs;
    //             ptrs.reserve(countries_.size());
    //             for (auto& c : countries_) ptrs.push_back(c.get());
    //             for (auto* c : ptrs)
    //                 if (c->action == -1)
    //                     c->select_action(ptrs, p_.p_war, p_.p_trade, p_.p_motion, p_.p_epidemic);
    //         }
    //
    //         int cnt_motion=0, cnt_trade=0, cnt_war=0, cnt_epi=0;
    //         double q_max_term = (1.0 - (double)ti / p_.tmax) * p_.max_mutation;
    //
    //         for (auto& c : countries_) {
    //             int act = c->action;
    //             if (act == 0) {
    //                 ++cnt_motion;
    //                 c->do_motion();
    //             } else if (act == 1 && c->ally != nullptr) {
    //                 ++cnt_trade;
    //                 Country::do_trade(*c, *c->ally, p_.k);
    //             } else if (act == 2 && c->enemy != nullptr) {
    //                 ++cnt_war;
    //                 Country::do_war(*c, *c->enemy, p_.l);
    //             } else if (act == 3) {
    //                 ++cnt_epi;
    //                 double pmax = (c->itype == IndividualType::Gray)
    //                                 ? q_max_term      // это (1 - t/tmax) * q_max, без -n_ep и без max(1,...)
    //                                 : p_.p_max;
    //                 c->do_epidemic(p_.ep_elite, p_.ep_dead, pmax);
    //             }
    //         }
    //
    //         _remove_empty();
    //         if (countries_.empty()) break;
    //
    //         std::sort(countries_.begin(), countries_.end(),
    //                   [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });
    //
    //         double f_min = countries_.front()->avg_f();
    //         double f_max = countries_.back()->avg_f();
    //
    //         if (f_min == f_max) {
    //             std::sort(countries_.begin(), countries_.end(),
    //                       [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });
    //             best_x = countries_[0]->population[0]->real_x();
    //             best_f = countries_[0]->population[0]->f_value;
    //             if (p_.printing) std::cout << "fmin == fmax\n";
    //             return {best_x, best_f, ti};
    //         }
    //
    //         std::vector<std::shared_ptr<Individual>> e_individuals;
    //         for (auto& c : countries_) {
    //             if (c->size() == 1) {
    //                 e_individuals.push_back(c->population[0]);
    //                 continue;
    //             }
    //             c->reproduction(p_.n_min, p_.n_max, p_.p_min, p_.p_max,
    //                             f_min, f_max, ti, p_.tmax);
    //             c->extinction(p_.m_min, p_.m_max, f_min, f_max);
    //         }
    //
    //         _remove_empty();
    //
    //         if (!countries_.empty()) {
    //             for (auto& ind : e_individuals)
    //                 _add_individual_to_random_country(ind);
    //         }
    //
    //         for (auto& c : countries_) c->truncate(2 * p_.N);
    //
    //         std::sort(countries_.begin(), countries_.end(),
    //                   [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });
    //
    //         if (countries_.empty()) break;
    //
    //         best_x = countries_[0]->population[0]->real_x();
    //         best_f = countries_[0]->population[0]->f_value;
    //
    //         if (p_.printing) {
    //             std::cout << ti << ") f=" << best_f << std::endl;
    //         }
    //
    //         // 1) Остановка по Евклидову расстоянию x до оптимума (epsilon)
    //         double dist = 0.0;
    //         for (size_t i = 0; i < best_x.size(); ++i) {
    //             dist += (best_x[i] - canonical_x[i]) * (best_x[i] - canonical_x[i]);
    //         }
    //         dist = std::sqrt(dist);
    //
    //         // if (dist <= epsilon) {
    //         //     return {best_x, best_f, ti};
    //         // }
    //
    //         // 2) Остановка по близости f(x) к f(canonical_x) (y_epsilon)
    //         // if (y_epsilon.has_value() && std::abs(best_f - canonical_y) <= y_epsilon.value()) {
    //         //     return {best_x, best_f, ti};
    //         // }
    //         //
    //         // if (best_f < canonical_y) {
    //         //     return {best_x, best_f, ti};
    //         // }
    //
    //         // 3) Остановка по лимиту вызовов внутри итерации
    //         if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
    //             if (p_.printing) std::cout << "Лимит вызовов функции достигнут\n";
    //             return {best_x, best_f, ti};
    //         }
    //     }
    //
    //     return {best_x, best_f, ti > p_.tmax ? p_.tmax : ti};
    // }

private:
    FuncT f_;
    Params p_;
    std::shared_ptr<long> calls_count_;
    std::vector<std::unique_ptr<Country>> countries_;
    ActionAdaptation action_adapt_;

    void _remove_empty() {
        countries_.erase(
            std::remove_if(countries_.begin(), countries_.end(),
                           [](const auto& c) { return c->empty(); }),
            countries_.end());
    }

    void _add_individual_to_random_country(const std::shared_ptr<Individual>& ind) {
        if (countries_.empty()) return;
        auto& rc  = countries_[rand_int(0, (int)countries_.size() - 1)];
        int   dim = (int)rc->x_min.size();
        std::shared_ptr<Individual> converted;
        if (ind->itype == rc->itype) {
            converted = ind->clone();
        } else if (rc->itype == IndividualType::Real) {
            auto rx   = ind->real_x();
            auto ni   = std::make_shared<RealIndividual>(rx, rc->x_min, rc->x_max, ind->f_value);
            ni->ep_n  = ind->ep_n;
            converted = ni;
        } else {
            auto rx = ind->real_x();
            std::vector<uint64_t> gc(dim);
            for (int d = 0; d < dim; ++d) {
                double   step = (rc->x_max[d] - rc->x_min[d]) / (double)((1ULL << rc->genes[d]) - 1);
                int64_t  v    = (int64_t)std::round((rx[d] - rc->x_min[d]) / step);
                uint64_t maxv = (1ULL << rc->genes[d]) - 1;
                gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)maxv));
            }
            auto ni  = std::make_shared<GrayIndividual>(gc, rc->x_min, rc->x_max, rc->genes, ind->f_value);
            ni->ep_n = ind->ep_n;
            converted = ni;
        }
        rc->population.push_back(converted);
        rc->sort_population();
    }
};
