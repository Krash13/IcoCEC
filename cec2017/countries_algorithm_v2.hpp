// countries_algorithm.hpp
// Полностью обновленная и исправленная версия алгоритма ICO (CountriesAlgorithm)
// Полная совместимость с main_cec2017.cpp, benchmark.hpp и CEC2017 suite.

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
#include <array>

// ============================================================================
// Gray code utilities
// ============================================================================

static inline uint64_t tc_to_gray_code(uint64_t n) noexcept {
    return n ^ (n >> 1);
}

static inline uint64_t gray_code_to_tc(uint64_t g) noexcept {
    uint64_t n = 0;
    for (; g; g >>= 1) {
        n ^= g;
    }
    return n;
}

// ============================================================================
// RNG utilities
// ============================================================================

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

// ============================================================================
// 5 Actions Choice:
// 0: Motion, 1: Trade, 2: War, 3: Epidemic, 4: Migration
// ============================================================================

static inline int weighted_action_choice(double pmotion, double ptrade, double pwar, double pepidemic, double pmigration) {
    double probs[5] = {pmotion, ptrade, pwar, pepidemic, pmigration};
    double sum = probs[0] + probs[1] + probs[2] + probs[3] + probs[4];
    if (sum <= 0.0) return rand_int(0, 4);

    double r = rand_uniform(0.0, sum);
    double acc = 0.0;
    for (int i = 0; i < 5; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return 4;
}

// ============================================================================
// Individual Types & Base Individual
// ============================================================================

enum class IndividualType { Gray, Real };

using FuncT = std::function<double(const std::vector<double>&)>;

struct Individual {
    std::vector<double> xmin, xmax;
    double fvalue = std::numeric_limits<double>::infinity();
    int epn = 0;
    IndividualType itype;

    Individual(std::vector<double> xmin, std::vector<double> xmax, IndividualType t)
        : xmin(std::move(xmin)), xmax(std::move(xmax)), itype(t) {}

    virtual ~Individual() = default;
    virtual std::vector<double> real_x() const = 0;
    virtual std::shared_ptr<Individual> clone() const = 0;

    bool operator<(const Individual& o) const noexcept { return fvalue < o.fvalue; }
    bool operator>(const Individual& o) const noexcept { return fvalue > o.fvalue; }
    bool operator<=(const Individual& o) const noexcept { return fvalue <= o.fvalue; }
};

// ============================================================================
// Gray Individual
// ============================================================================

struct GrayIndividual : Individual {
    std::vector<int> genes;
    std::vector<uint64_t> code;
    std::vector<double> steps;

    GrayIndividual(std::vector<uint64_t> gray_code,
                   const std::vector<double>& xmin,
                   const std::vector<double>& xmax,
                   const std::vector<int>& genes,
                   const FuncT& func)
        : Individual(xmin, xmax, IndividualType::Gray),
          genes(genes),
          code(std::move(gray_code)) {
        init_steps();
        fvalue = func(real_x());
    }

    GrayIndividual(std::vector<uint64_t> gray_code,
                   const std::vector<double>& xmin,
                   const std::vector<double>& xmax,
                   const std::vector<int>& genes,
                   double fval)
        : Individual(xmin, xmax, IndividualType::Gray),
          genes(genes),
          code(std::move(gray_code)) {
        init_steps();
        fvalue = fval;
    }

    std::vector<double> real_x() const override {
        int dim = (int)genes.size();
        std::vector<double> rx(dim);
        for (int i = 0; i < dim; ++i) {
            rx[i] = xmin[i] + steps[i] * (double)gray_code_to_tc(code[i]);
        }
        return rx;
    }

    std::vector<uint64_t> decimal_x() const {
        int dim = (int)genes.size();
        std::vector<uint64_t> dec(dim);
        for (int i = 0; i < dim; ++i) {
            dec[i] = gray_code_to_tc(code[i]);
        }
        return dec;
    }

    static std::shared_ptr<GrayIndividual> from_decimal(
        const std::vector<uint64_t>& decimal,
        const std::vector<double>& xmin,
        const std::vector<double>& xmax,
        const std::vector<int>& genes,
        const FuncT& func) {
        int dim = (int)genes.size();
        std::vector<uint64_t> gc(dim);
        for (int i = 0; i < dim; ++i) {
            uint64_t max_val = (1ULL << genes[i]) - 1;
            gc[i] = tc_to_gray_code(std::min(decimal[i], max_val));
        }
        return std::make_shared<GrayIndividual>(gc, xmin, xmax, genes, func);
    }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<GrayIndividual>(code, xmin, xmax, genes, fvalue);
        p->epn = epn;
        return p;
    }

    std::shared_ptr<GrayIndividual> mutate(double qmax_term, const FuncT& func) const {
        int n = std::max(0, (int)std::floor(qmax_term - epn));
        if (n == 0) {
            auto p = std::make_shared<GrayIndividual>(code, xmin, xmax, genes, fvalue);
            return p;
        }

        int total_bits = 0;
        for (int g : genes) total_bits += g;

        std::vector<uint8_t> bits(total_bits);
        int pos = 0;
        for (int i = 0; i < (int)genes.size(); ++i) {
            for (int b = genes[i] - 1; b >= 0; --b) {
                bits[pos++] = (uint8_t)((code[i] >> b) & 1);
            }
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
        pos = 0;
        for (int i = 0; i < (int)genes.size(); ++i) {
            uint64_t v = 0;
            for (int b = genes[i] - 1; b >= 0; --b) {
                v = (v << 1) | (uint64_t)bits[pos++];
            }
            new_code[i] = v;
        }

        auto p = std::make_shared<GrayIndividual>(new_code, xmin, xmax, genes, func);
        return p;
    }

    static std::pair<std::shared_ptr<GrayIndividual>, std::shared_ptr<GrayIndividual>>
    crossover(const GrayIndividual& a, const GrayIndividual& b, const FuncT& func) {
        int total_bits = 0;
        for (int g : a.genes) total_bits += g;

        auto to_bits = [&](const GrayIndividual& ind) {
            std::vector<uint8_t> bits(total_bits);
            int pos = 0;
            for (int i = 0; i < (int)ind.genes.size(); ++i) {
                for (int bit = ind.genes[i] - 1; bit >= 0; --bit) {
                    bits[pos++] = (uint8_t)((ind.code[i] >> bit) & 1);
                }
            }
            return bits;
        };

        auto bits_a = to_bits(a);
        auto bits_b = to_bits(b);
        std::vector<uint8_t> nb1(total_bits), nb2(total_bits);

        if (rand_uniform(0.0, 1.0) < 0.7) {
            for (int i = 0; i < total_bits; ++i) {
                if (rand_int(0, 1) == 1) {
                    nb1[i] = bits_b[i];
                    nb2[i] = bits_a[i];
                } else {
                    nb1[i] = bits_a[i];
                    nb2[i] = bits_b[i];
                }
            }
        } else {
            int k1, k2;
            if (total_bits <= 2) {
                k1 = 0;
                k2 = total_bits;
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
                for (int bit = a.genes[i] - 1; bit >= 0; --bit) {
                    v = (v << 1) | (uint64_t)bits[pos++];
                }
                gc[i] = v;
            }
            return std::make_shared<GrayIndividual>(gc, a.xmin, a.xmax, a.genes, func);
        };

        return {bits_to_ind(nb1), bits_to_ind(nb2)};
    }

private:
    void init_steps() {
        int dim = (int)genes.size();
        steps.resize(dim);
        for (int i = 0; i < dim; ++i) {
            steps[i] = (xmax[i] - xmin[i]) / (double)((1ULL << genes[i]) - 1);
        }
    }
};

// ============================================================================
// Real Individual
// ============================================================================

struct RealIndividual : Individual {
    std::vector<double> x;

    RealIndividual(std::vector<double> x,
                   const std::vector<double>& xmin,
                   const std::vector<double>& xmax,
                   const FuncT& func)
        : Individual(xmin, xmax, IndividualType::Real), x(std::move(x)) {
        fvalue = func(this->x);
    }

    RealIndividual(std::vector<double> x,
                   const std::vector<double>& xmin,
                   const std::vector<double>& xmax,
                   double fval)
        : Individual(xmin, xmax, IndividualType::Real), x(std::move(x)) {
        fvalue = fval;
    }

    std::vector<double> real_x() const override { return x; }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<RealIndividual>(x, xmin, xmax, fvalue);
        p->epn = epn;
        return p;
    }

    void update_f(const FuncT& func) {
        fvalue = func(x);
    }

    void mutation(const FuncT& func, double pmax) {
        epn += 1;
        int dim = (int)x.size();
        for (int i = 0; i < dim; ++i) {
            double range = xmax[i] - xmin[i];
            double perturb = pmax * rand_uniform(-0.5, 0.5) * range / (double)epn;
            x[i] = std::clamp(x[i] + perturb, xmin[i], xmax[i]);
        }
        update_f(func);
    }

    static std::shared_ptr<RealIndividual> crossover(
        const RealIndividual& a, const RealIndividual& b,
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
            u(i) = rand_uniform(0.0, 1.0);
        }

        Eigen::ArrayXd child = lower + u * (upper - lower);
        child = child.max(min_x).min(max_x);

        std::vector<double> new_x(child.data(), child.data() + child.size());
        return std::make_shared<RealIndividual>(std::move(new_x), xmin, xmax, func);
    }
};

// ============================================================================
// Country
// ============================================================================

struct Country {
    std::vector<std::shared_ptr<Individual>> population;
    std::vector<double> xmin, xmax;
    std::vector<int> genes;
    FuncT f;
    int N;
    IndividualType itype;

    int action = -1;
    Country* ally = nullptr;
    Country* enemy = nullptr;

    Country(int N,
            const std::vector<double>& xmin,
            const std::vector<double>& xmax,
            const FuncT& func,
            IndividualType it,
            const std::vector<int>& genes)
        : xmin(xmin), xmax(xmax), genes(genes), f(func), N(N), itype(it) {
        int dim = (int)xmin.size();
        population.reserve(N);

        if (itype == IndividualType::Gray) {
            std::vector<uint64_t> lmin(dim), lmax(dim);
            for (int d = 0; d < dim; ++d) {
                uint64_t max_val = (1ULL << genes[d]) - 1;
                lmin[d] = rand_uint64(0, max_val - 1);
                lmax[d] = rand_uint64(lmin[d] + 1, max_val);
            }
            for (int i = 0; i < N; ++i) {
                std::vector<uint64_t> dec(dim);
                for (int d = 0; d < dim; ++d) {
                    dec[d] = rand_uint64(lmin[d], lmax[d]);
                }
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
                for (int d = 0; d < dim; ++d) {
                    x[d] = rand_uniform(lmin[d], lmax[d]);
                }
                population.push_back(std::make_shared<RealIndividual>(std::move(x), xmin, xmax, f));
            }
        }
        sort_population();
    }

    bool empty() const noexcept { return population.empty(); }
    int size() const noexcept { return (int)population.size(); }

    void sort_population() {
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a->fvalue < b->fvalue; });
    }

    double best_f() const noexcept {
        return population.empty() ? std::numeric_limits<double>::infinity() : population[0]->fvalue;
    }

    double avg_f() const noexcept {
        if (population.empty()) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (const auto& ind : population) s += ind->fvalue;
        return s / (double)population.size();
    }

    void select_action(std::vector<Country*>& all_countries,
                       double pmotion, double ptrade, double pwar, double pepidemic, double pmigration) {
        action = weighted_action_choice(pmotion, ptrade, pwar, pepidemic, pmigration);

        if (action == 1) {
            std::vector<Country*> candidates;
            for (auto* c : all_countries) {
                if (c->action == -1 && c != this) candidates.push_back(c);
            }
            if (!candidates.empty()) {
                auto chosen = candidates[rand_int(0, (int)candidates.size() - 1)];
                ally = chosen;
                chosen->action = 1;
                chosen->ally = this;
            } else {
                int fallback = rand_int(0, 2);
                action = (fallback == 0) ? 0 : (fallback == 1 ? 3 : 4);
            }
        } else if (action == 2) {
            std::vector<Country*> candidates;
            for (auto* c : all_countries) {
                if (c->action == -1 && c != this) candidates.push_back(c);
            }
            if (!candidates.empty()) {
                auto chosen = candidates[rand_int(0, (int)candidates.size() - 1)];
                enemy = chosen;
                chosen->action = 2;
                chosen->enemy = this;
            } else {
                int fallback = rand_int(0, 2);
                action = (fallback == 0) ? 0 : (fallback == 1 ? 3 : 4);
            }
        }
    }

    void do_motion(double rmax = 2.0) {
        if (population.empty()) { action = -1; return; }

        if (itype == IndividualType::Gray) {
            auto best_dec = std::static_pointer_cast<GrayIndividual>(population[0])->decimal_x();
            for (int i = 1; i < size(); ++i) {
                auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
                auto dec = ind->decimal_x();
                double r = rand_uniform(0.0, rmax);
                std::vector<uint64_t> new_dec(genes.size());
                for (int d = 0; d < (int)genes.size(); ++d) {
                    int64_t diff = static_cast<int64_t>(best_dec[d]) - static_cast<int64_t>(dec[d]);
                    int64_t nd = static_cast<int64_t>(dec[d]) + static_cast<int64_t>(r * static_cast<double>(diff));
                    uint64_t max_val = (1ULL << genes[d]) - 1;
                    new_dec[d] = static_cast<uint64_t>(std::clamp(nd, (int64_t)0, (int64_t)max_val));
                }
                population[i] = GrayIndividual::from_decimal(new_dec, xmin, xmax, genes, f);
            }
        } else {
            const auto& best_x = std::static_pointer_cast<RealIndividual>(population[0])->x;
            for (int i = 1; i < size(); ++i) {
                auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
                double r = rand_uniform(0.0, rmax);
                for (int d = 0; d < (int)ind->x.size(); ++d) {
                    ind->x[d] = std::clamp(ind->x[d] + r * (best_x[d] - ind->x[d]), xmin[d], xmax[d]);
                }
                ind->update_f(f);
            }
        }
        sort_population();
        action = -1;
    }

    void do_epidemic(double elite_frac, double dead_frac, double pmax_real, double qmax_term_gray) {
        int n = size();
        int n_elite = (int)std::ceil(elite_frac * n);
        int n_dead  = (int)std::ceil(dead_frac * n);

        if (n_dead >= n) {
            population.clear();
            action = -1;
            return;
        }

        if (n_dead > 0) {
            population.erase(population.end() - n_dead, population.end());
        }

        for (int i = n_elite; i < size(); ++i) {
            if (itype == IndividualType::Gray) {
                auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
                auto new_ind = ind->mutate(qmax_term_gray, f);
                new_ind->epn = ind->epn + 1;
                population[i] = new_ind;
            } else {
                auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
                ind->mutation(f, pmax_real);
            }
        }
        sort_population();
        action = -1;
    }

    void do_migration(double migrate_frac = 0.3) {
        int n = size();
        if (n <= 1) { action = -1; return; }

        int n_migrate = std::clamp((int)std::ceil(migrate_frac * n), 1, n - 1);
        population.erase(population.end() - n_migrate, population.end());

        for (int i = 0; i < n_migrate; ++i) {
            population.push_back(make_random_individual());
        }
        sort_population();
        action = -1;
    }

    void reproduction(int nmin, int nmax, double pmin, double pmax,
                      double fmin, double fmax, int ti, int tmax) {
        if (size() < 2) return;
        double avg = avg_f();
        double n_frac = (fmax - avg) / (fmax - fmin + 1e-15);
        int n = std::clamp((int)std::ceil((nmax - nmin) * n_frac + nmin), nmin, nmax);

        if (itype == IndividualType::Gray) {
            for (int i = 0; i < n; ++i) {
                int k1 = rand_int(0, size() - 1);
                int k2 = k1;
                while (k2 == k1) k2 = rand_int(0, size() - 1);
                auto a = std::static_pointer_cast<GrayIndividual>(population[k1]);
                auto b = std::static_pointer_cast<GrayIndividual>(population[k2]);
                auto [c1, c2] = GrayIndividual::crossover(*a, *b, f);
                population.push_back(std::move(c1));
                population.push_back(std::move(c2));
            }
        } else {
            double p = std::clamp(
                pmax - (pmax - pmin) * (1.0 - (double)ti / tmax) * ((avg - fmin) / (fmax - fmin + 1e-15)),
                pmin, pmax
            );
            for (int i = 0; i < 2 * n; ++i) {
                int k1 = rand_int(0, size() - 1);
                int k2 = k1;
                while (k2 == k1) k2 = rand_int(0, size() - 1);
                auto a = std::static_pointer_cast<RealIndividual>(population[k1]);
                auto b = std::static_pointer_cast<RealIndividual>(population[k2]);
                population.push_back(RealIndividual::crossover(*a, *b, p, xmin, xmax, f));
            }
        }
        sort_population();
    }

    void extinction(int mmin, int mmax, double fmin, double fmax) {
        double avg = avg_f();
        int m = std::clamp(
            (int)((mmax - mmin) * ((avg - fmin) / (fmax - fmin + 1e-15)) + mmin),
            mmin, mmax
        );
        if (m >= size()) {
            population.clear();
            return;
        }
        population.erase(population.end() - m, population.end());
    }

    void truncate(int max_size) {
        if (size() > max_size) {
            population.resize(max_size);
        }
    }

    void update_individual_type() {
        int dim = (int)xmin.size();
        for (auto& ind : population) {
            if (ind->itype == itype) continue;
            if (itype == IndividualType::Real) {
                auto rx = ind->real_x();
                auto ni = std::make_shared<RealIndividual>(std::move(rx), xmin, xmax, ind->fvalue);
                ni->epn = ind->epn;
                ind = ni;
            } else {
                auto rx = ind->real_x();
                std::vector<uint64_t> gc(dim);
                for (int d = 0; d < dim; ++d) {
                    double step = (xmax[d] - xmin[d]) / (double)((1ULL << genes[d]) - 1);
                    int64_t v = (int64_t)std::round((rx[d] - xmin[d]) / step);
                    uint64_t max_v = (1ULL << genes[d]) - 1;
                    gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)max_v));
                }
                auto ni = std::make_shared<GrayIndividual>(gc, xmin, xmax, genes, ind->fvalue);
                ni->epn = ind->epn;
                ind = ni;
            }
        }
    }

    std::shared_ptr<Individual> make_random_individual() const {
        const int dim = static_cast<int>(xmin.size());
        if (itype == IndividualType::Gray) {
            std::vector<uint64_t> decimal(dim);
            for (int d = 0; d < dim; ++d) {
                const uint64_t max_val = (1ULL << genes[d]) - 1;
                decimal[d] = rand_uint64(0, max_val);
            }
            return GrayIndividual::from_decimal(decimal, xmin, xmax, genes, f);
        }
        std::vector<double> x(dim);
        for (int d = 0; d < dim; ++d) {
            x[d] = rand_uniform(xmin[d], xmax[d]);
        }
        return std::make_shared<RealIndividual>(std::move(x), xmin, xmax, f);
    }

    static void do_trade(Country& c1, Country& c2, int k) {
        int actual_k = k;
        if (c1.size() <= k || c2.size() <= k) {
            actual_k = std::min(c1.size(), c2.size()) / 2;
        }
        if (actual_k <= 0) {
            c1.action = -1; c2.action = -1;
            c1.ally = nullptr; c2.ally = nullptr;
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

        auto remove_by_idx = [](std::vector<std::shared_ptr<Individual>>& pop, const std::vector<int>& idx_in) {
            std::vector<int> idx = idx_in;
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

        c1.update_individual_type();
        c2.update_individual_type();
        c1.sort_population();
        c2.sort_population();

        c1.action = -1; c2.action = -1;
        c1.ally = nullptr; c2.ally = nullptr;
    }

    static std::shared_ptr<Individual> recruit_from_duel(
        const Individual& winner, const Individual& loser,
        Country& home, double rmax) {
        auto wx = winner.real_x();
        auto lx = loser.real_x();
        const int dim = static_cast<int>(wx.size());

        double dist2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = wx[d] - lx[d];
            dist2 += diff * diff;
        }
        const double dist = std::sqrt(dist2);

        std::vector<double> noise(dim);
        double nrm2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            noise[d] = rand_uniform(-1.0, 1.0);
            nrm2 += noise[d] * noise[d];
        }
        const double nrm = std::sqrt(nrm2) + 1e-12;

        const double alpha = rand_uniform(0.1, std::max(0.15, rmax * 0.4));
        const double jitter = 0.05 * dist;

        std::vector<double> nx(dim);
        for (int d = 0; d < dim; ++d) {
            const double reflected = wx[d] + alpha * (wx[d] - lx[d]);
            nx[d] = std::clamp(reflected + jitter * (noise[d] / nrm), home.xmin[d], home.xmax[d]);
        }

        if (home.itype == IndividualType::Real) {
            return std::make_shared<RealIndividual>(std::move(nx), home.xmin, home.xmax, home.f);
        }

        std::vector<uint64_t> dec(dim);
        for (int d = 0; d < dim; ++d) {
            const double step = (home.xmax[d] - home.xmin[d]) / static_cast<double>((1ULL << home.genes[d]) - 1);
            const int64_t v = static_cast<int64_t>(std::round((nx[d] - home.xmin[d]) / step));
            const uint64_t max_v = (1ULL << home.genes[d]) - 1;
            dec[d] = static_cast<uint64_t>(std::clamp(v, (int64_t)0, static_cast<int64_t>(max_v)));
        }
        return GrayIndividual::from_decimal(dec, home.xmin, home.xmax, home.genes, home.f);
    }

    static void do_war(Country& c1, Country& c2, int l, double rmax = 2.0) {
        int actual_l = l;
        if (c1.size() <= l || c2.size() <= l) {
            actual_l = std::min(c1.size(), c2.size());
        }
        if (actual_l <= 0) {
            c1.action = -1; c2.action = -1;
            c1.enemy = nullptr; c2.enemy = nullptr;
            return;
        }

        auto pick_indices = [](int sz, int cnt) {
            std::vector<int> idx(sz);
            std::iota(idx.begin(), idx.end(), 0);
            for (int i = 0; i < cnt; ++i) {
                std::swap(idx[i], idx[rand_int(i, sz - 1)]);
            }
            idx.resize(cnt);
            return idx;
        };

        auto remove_by_idx = [](std::vector<std::shared_ptr<Individual>>& pop, const std::vector<int>& idx_in) {
            std::vector<int> idx = idx_in;
            std::sort(idx.begin(), idx.end(), std::greater<int>());
            for (int i : idx) pop.erase(pop.begin() + i);
        };

        const auto idx1 = pick_indices(c1.size(), actual_l);
        const auto idx2 = pick_indices(c2.size(), actual_l);

        std::vector<std::shared_ptr<Individual>> war1, war2;
        war1.reserve(actual_l);
        war2.reserve(actual_l);
        for (int i : idx1) war1.push_back(c1.population[i]->clone());
        for (int i : idx2) war2.push_back(c2.population[i]->clone());

        remove_by_idx(c1.population, idx1);
        remove_by_idx(c2.population, idx2);

        int wins1 = 0, wins2 = 0;
        std::vector<std::shared_ptr<Individual>> survivors1, survivors2;
        survivors1.reserve(actual_l);
        survivors2.reserve(actual_l);

        for (int i = 0; i < actual_l; ++i) {
            if (*war1[i] < *war2[i]) {
                wins1++;
                survivors1.push_back(war1[i]);
                c2.population.push_back(recruit_from_duel(*war1[i], *war2[i], c2, rmax));
            } else if (*war2[i] < *war1[i]) {
                wins2++;
                survivors2.push_back(war2[i]);
                c1.population.push_back(recruit_from_duel(*war2[i], *war1[i], c1, rmax));
            } else {
                survivors1.push_back(war1[i]);
                survivors2.push_back(war2[i]);
            }
        }

        auto assimilate = [](Country& winner, const std::shared_ptr<Individual>& prisoner) {
            if (winner.empty()) {
                winner.population.push_back(prisoner);
                return;
            }
            const auto capital = winner.population[0]->real_x();
            auto px = prisoner->real_x();
            const double r = rand_uniform(0.35, 0.85);
            for (int d = 0; d < (int)px.size(); ++d) {
                px[d] = std::clamp(px[d] + r * (capital[d] - px[d]), winner.xmin[d], winner.xmax[d]);
            }
            if (winner.itype == IndividualType::Real) {
                auto ni = std::make_shared<RealIndividual>(std::move(px), winner.xmin, winner.xmax, winner.f);
                ni->epn = prisoner->epn;
                winner.population.push_back(std::move(ni));
            } else {
                winner.population.push_back(prisoner);
            }
        };

        if (wins1 > wins2) {
            for (auto& w : survivors1) c1.population.push_back(w);
            for (auto& p : survivors2) assimilate(c1, p);
        } else if (wins2 > wins1) {
            for (auto& p : survivors1) assimilate(c2, p);
            for (auto& w : survivors2) c2.population.push_back(w);
        } else {
            for (auto& w : survivors1) c1.population.push_back(w);
            for (auto& w : survivors2) c2.population.push_back(w);
        }

        c1.update_individual_type();
        c2.update_individual_type();
        c1.sort_population();
        c2.sort_population();

        c1.action = -1; c2.action = -1;
        c1.enemy = nullptr; c2.enemy = nullptr;
    }
};

// ============================================================================
// CountriesAlgorithm Engine
// ============================================================================

class CountriesAlgorithm {
public:
    using Vec = std::vector<double>;

    struct Params {
        std::vector<double> xmin, xmax;
        std::vector<int> genes;
        double pmin = 0.1;
        double pmax = 0.5;
        int M = 10;
        int N = 20;
        int nmin = 1;
        int nmax = 5;
        int mmin = 1;
        int mmax = 3;
        int k = 3;
        int l = 3;
        double ep_elite = 0.2;
        double ep_dead = 0.3;
        int max_mutation = 3;
        int tmax = 1000;
        double gray_percent = 0.5;
        bool printing = true;

        // Поля 5 действий (основные и псевдонимы)
        double pmotion    = 0.25;
        double ptrade     = 0.20;
        double pwar       = 0.20;
        double pepidemic  = 0.20;
        double pmigration = 0.15;

        double p_motion    = 0.25;
        double p_trade     = 0.20;
        double p_war       = 0.20;
        double p_epidemic  = 0.20;
        double p_migration = 0.15;

        // Псевдонимы для совместимости с кодом с подчеркиваниями
        std::vector<double> x_min;
        std::vector<double> x_max;
        double p_min = 0.1;
        double p_max = 0.5;
        int n_min = 1;
        int n_max = 5;
        int m_min = 1;
        int m_max = 3;

        bool   adaptive_actions = true;
        double action_alpha     = 0.076;   // Скорость обучения EMA
        double action_pmin      = 0.05;   // Минимальный гарантированный порог
        double action_warmup_frac = 0.14; // Доля итераций warm-up

        // Настройки глобального периодического рестарта худших стран
        int    stagnation_limit     = 25;   // Число итераций без глобального улучшения для рестарта
        double restart_country_frac = 0.15; // Доля худших стран для полного пересоздания
        double migration_frac       = 0.30; // Доля особей, телепортируемых при действии Migration

        void sync_all_fields() {
            if (!x_min.empty()) xmin = x_min;
            else x_min = xmin;

            if (!x_max.empty()) xmax = x_max;
            else x_max = xmax;

            if (p_min != 0.1) pmin = p_min;
            else p_min = pmin;

            if (p_max != 0.5) pmax = p_max;
            else p_max = pmax;

            if (n_min != 1) nmin = n_min;
            else n_min = nmin;

            if (n_max != 5) nmax = n_max;
            else n_max = nmax;

            if (m_min != 1) mmin = m_min;
            else m_min = mmin;

            if (m_max != 3) mmax = m_max;
            else m_max = mmax;

            if (p_motion != 0.25 || p_trade != 0.20 || p_war != 0.20 || p_epidemic != 0.20 || p_migration != 0.15) {
                pmotion = p_motion;
                ptrade = p_trade;
                pwar = p_war;
                pepidemic = p_epidemic;
                pmigration = p_migration;
            } else {
                p_motion = pmotion;
                p_trade = ptrade;
                p_war = pwar;
                p_epidemic = pepidemic;
                p_migration = pmigration;
            }
        }
    };

    struct ActionAdaptation {
        // 0: motion, 1: trade, 2: war, 3: epidemic, 4: migration
        std::array<double, 5> reward = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<double, 5> probs  = {0.25, 0.20, 0.20, 0.20, 0.15};

        void update(int action_idx, double f_before, double f_after, long calls_spent, double alpha) {
            if (action_idx < 0 || action_idx >= 5) return;
            if (calls_spent <= 0) calls_spent = 1;
            double improvement = std::max(0.0, f_before - f_after);
            double credit = improvement / (double)calls_spent;
            reward[action_idx] = (1.0 - alpha) * reward[action_idx] + alpha * credit;
        }

        void renormalize(double p_min) {
            double sum = 0.0;
            for (double r : reward) sum += r;
            if (sum <= 1e-15) {
                probs = {0.25, 0.20, 0.20, 0.20, 0.15};
                return;
            }
            double residual = 1.0 - 5.0 * p_min;
            for (int i = 0; i < 5; ++i) {
                probs[i] = p_min + residual * (reward[i] / sum);
            }
        }
    };

    explicit CountriesAlgorithm(FuncT func, Params params)
        : p_(std::move(params)),
          calls_count_(std::make_shared<long>(0)) {
        p_.sync_all_fields();
        init_internal(std::move(func));
    }

    CountriesAlgorithm(FuncT func, const Vec& x_min, const Vec& x_max, Params params)
        : p_(std::move(params)),
          calls_count_(std::make_shared<long>(0)) {
        p_.xmin = x_min;
        p_.xmax = x_max;
        p_.x_min = x_min;
        p_.x_max = x_max;
        p_.sync_all_fields();
        init_internal(std::move(func));
    }

    void init_internal(FuncT func) {
        double sum_p = p_.pmotion + p_.ptrade + p_.pwar + p_.pepidemic + p_.pmigration;
        if (std::abs(sum_p - 1.0) > 1e-5) {
            p_.pmotion /= sum_p;
            p_.ptrade /= sum_p;
            p_.pwar /= sum_p;
            p_.pepidemic /= sum_p;
            p_.pmigration /= sum_p;
            p_.sync_all_fields();
        }

        auto counter = calls_count_;
        f_ = [counter, user_func = std::move(func)](const std::vector<double>& x) -> double {
            (*counter)++;
            return user_func(x);
        };

        if (p_.genes.empty()) {
            p_.genes.assign(p_.xmin.size(), 32);
        }

        init_countries();
    }

    void init_countries() {
        countries_.clear();
        int gray_M = (int)std::round(p_.gray_percent * p_.M);
        int real_M = p_.M - gray_M;
        countries_.reserve(p_.M);

        for (int i = 0; i < gray_M; ++i) {
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.xmin, p_.xmax, f_, IndividualType::Gray, p_.genes
            ));
        }
        for (int i = 0; i < real_M; ++i) {
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.xmin, p_.xmax, f_, IndividualType::Real, p_.genes
            ));
        }
    }

    std::tuple<Vec, double, long> start(
        const Vec& canonical_x, double epsilon,
        std::optional<double> y_epsilon = std::nullopt,
        std::optional<long> max_calls = std::nullopt) {

        double canonical_y = f_(canonical_x);
        Vec best_x;
        double best_f = std::numeric_limits<double>::infinity();
        long ti = 0;

        if (!countries_.empty() && !countries_[0]->population.empty()) {
            best_x = countries_[0]->population[0]->real_x();
            best_f = countries_[0]->population[0]->fvalue;
        }

        long warmup_iters = static_cast<long>(std::round(p_.action_warmup_frac * static_cast<double>(p_.tmax)));
        int iters_without_improvement = 0;

        for (ti = 1; ti <= p_.tmax; ++ti) {
            double progress = static_cast<double>(ti - 1) / std::max(1, p_.tmax - 1);
            double rmax = 2.0 - 0.8 * std::pow(progress, 0.6);

            if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
                if (p_.printing) std::cout << "Max calls reached: " << *calls_count_ << std::endl;
                return {best_x, best_f, ti};
            }

            if (countries_.size() == 1) {
                split_single_country();
            }

            std::vector<Country*> ptrs;
            ptrs.reserve(countries_.size());
            for (auto& c : countries_) ptrs.push_back(c.get());

            for (auto* c : ptrs) {
                if (c->action == -1) {
                    c->select_action(ptrs, p_.pmotion, p_.ptrade, p_.pwar, p_.pepidemic, p_.pmigration);
                }
            }

            double qmax_term = (1.0 - (double)ti / p_.tmax) * p_.max_mutation;

            std::vector<double> avg_before(countries_.size());
            for (size_t i = 0; i < countries_.size(); ++i) {
                avg_before[i] = countries_[i]->avg_f();
            }

            for (size_t i = 0; i < countries_.size(); ++i) {
                auto& c = countries_[i];
                int act = c->action;
                long calls_before = *calls_count_;

                if (act == 0) {
                    c->do_motion(rmax);
                } else if (act == 1 && c->ally != nullptr) {
                    Country::do_trade(*c, *c->ally, p_.k);
                } else if (act == 2 && c->enemy != nullptr) {
                    Country::do_war(*c, *c->enemy, p_.l, rmax);
                } else if (act == 3) {
                    double pmax = (c->itype == IndividualType::Gray) ? qmax_term : p_.pmax;
                    c->do_epidemic(p_.ep_elite, p_.ep_dead, pmax, qmax_term);
                } else if (act == 4) {
                    c->do_migration(p_.migration_frac);
                }

                if (p_.adaptive_actions && act != -1) {
                    long calls_after = *calls_count_;
                    double f_after = c->avg_f();
                    action_adapt_.update(act, avg_before[i], f_after, calls_after - calls_before, p_.action_alpha);
                }
            }

            if (p_.adaptive_actions && ti >= warmup_iters) {
                action_adapt_.renormalize(p_.action_pmin);
                p_.pmotion    = action_adapt_.probs[0];
                p_.ptrade     = action_adapt_.probs[1];
                p_.pwar       = action_adapt_.probs[2];
                p_.pepidemic  = action_adapt_.probs[3];
                p_.pmigration = action_adapt_.probs[4];
                p_.sync_all_fields();
            }

            remove_empty();
            if (countries_.empty()) break;

            std::sort(countries_.begin(), countries_.end(),
                      [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });

            double fmin = countries_.front()->avg_f();
            double fmax = countries_.back()->avg_f();

            if (fmin == fmax && countries_.size() > 1) {
                restart_stagnant_countries(0.5);
                std::sort(countries_.begin(), countries_.end(),
                          [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });
                fmin = countries_.front()->avg_f();
                fmax = countries_.back()->avg_f();
            }

            std::vector<std::shared_ptr<Individual>> e_individuals;
            for (auto& c : countries_) {
                if (c->size() <= 1) {
                    if (c->size() == 1) e_individuals.push_back(c->population[0]);
                    continue;
                }
                c->reproduction(p_.nmin, p_.nmax, p_.pmin, p_.pmax, fmin, fmax, (int)ti, p_.tmax);
                c->extinction(p_.mmin, p_.mmax, fmin, fmax);
            }

            remove_empty();

            if (!countries_.empty()) {
                for (const auto& ind : e_individuals) {
                    add_individual_to_random_country(ind);
                }
                for (auto& c : countries_) {
                    c->truncate(2 * p_.N);
                }
            }

            std::sort(countries_.begin(), countries_.end(),
                      [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

            if (countries_.empty()) break;

            if (countries_[0]->population[0]->fvalue < best_f) {
                best_f = countries_[0]->population[0]->fvalue;
                best_x = countries_[0]->population[0]->real_x();
                iters_without_improvement = 0;
            } else {
                iters_without_improvement++;
            }

            if (iters_without_improvement >= p_.stagnation_limit && countries_.size() > 1) {
                restart_stagnant_countries(p_.restart_country_frac);
                iters_without_improvement = 0;
            }

            if (p_.printing && ti % 50 == 0) {
                std::cout << "Iter: " << ti << ", Best F: " << best_f << ", Calls: " << *calls_count_ << std::endl;
            }

            double dist = 0.0;
            for (size_t i = 0; i < best_x.size(); ++i) {
                double diff = best_x[i] - canonical_x[i];
                dist += diff * diff;
            }
            // if (std::sqrt(dist) <= epsilon) {
            //     return {best_x, best_f, ti};
            // }

            // if (y_epsilon.has_value() && std::abs(best_f - canonical_y) <= y_epsilon.value()) {
            //     return {best_x, best_f, ti};
            // }
            // if (best_f <= canonical_y) {
            //     return {best_x, best_f, ti};
            // }
        }

        return {best_x, best_f, ti};
    }

private:
    FuncT f_;
    Params p_;
    std::shared_ptr<long> calls_count_;
    std::vector<std::unique_ptr<Country>> countries_;
    ActionAdaptation action_adapt_;

    void remove_empty() {
        countries_.erase(
            std::remove_if(countries_.begin(), countries_.end(),
                           [](const auto& c) { return c->empty(); }),
            countries_.end()
        );
    }

    void add_individual_to_random_country(const std::shared_ptr<Individual>& ind) {
        if (countries_.empty()) return;
        auto& rc = countries_[rand_int(0, (int)countries_.size() - 1)];
        int dim = (int)rc->xmin.size();
        std::shared_ptr<Individual> converted;

        if (ind->itype == rc->itype) {
            converted = ind->clone();
        } else if (rc->itype == IndividualType::Real) {
            auto rx = ind->real_x();
            auto ni = std::make_shared<RealIndividual>(std::move(rx), rc->xmin, rc->xmax, ind->fvalue);
            ni->epn = ind->epn;
            converted = ni;
        } else {
            auto rx = ind->real_x();
            std::vector<uint64_t> gc(dim);
            for (int d = 0; d < dim; ++d) {
                double step = (rc->xmax[d] - rc->xmin[d]) / (double)((1ULL << rc->genes[d]) - 1);
                int64_t v = (int64_t)std::round((rx[d] - rc->xmin[d]) / step);
                uint64_t max_v = (1ULL << rc->genes[d]) - 1;
                gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)max_v));
            }
            auto ni = std::make_shared<GrayIndividual>(gc, rc->xmin, rc->xmax, rc->genes, ind->fvalue);
            ni->epn = ind->epn;
            converted = ni;
        }
        rc->population.push_back(converted);
        rc->sort_population();
    }

    void restart_stagnant_countries(double frac) {
        if (countries_.size() <= 1) return;
        int num_restart = std::clamp((int)std::ceil(frac * (double)countries_.size()), 1, (int)countries_.size() - 1);

        size_t start_idx = countries_.size() - num_restart;
        for (size_t i = start_idx; i < countries_.size(); ++i) {
            auto& c = countries_[i];
            int target_n = std::max(c->size(), p_.N);
            c->population.clear();
            c->population.reserve(target_n);

            for (int j = 0; j < target_n; ++j) {
                c->population.push_back(c->make_random_individual());
            }
            c->sort_population();
            c->action = -1;
            c->ally = nullptr;
            c->enemy = nullptr;
        }
    }

    void split_single_country() {
        if (countries_.size() != 1) return;

        auto& original = countries_[0];
        auto all_individuals = std::move(original->population);
        countries_.clear();

        int total_inds = (int)all_individuals.size();
        int target_size = std::max(2, p_.N / 2);
        int num_new_countries = std::max(2, (total_inds + target_size - 1) / target_size);

        std::shuffle(all_individuals.begin(), all_individuals.end(), rng_engine);

        int num_gray = (int)std::round(p_.gray_percent * (double)num_new_countries);
        num_gray = std::clamp(num_gray, 0, num_new_countries);

        int ind_offset = 0;
        for (int c_idx = 0; c_idx < num_new_countries; ++c_idx) {
            IndividualType it = (c_idx < num_gray) ? IndividualType::Gray : IndividualType::Real;
            auto new_c = std::make_unique<Country>(
                target_size, p_.xmin, p_.xmax, f_, it, p_.genes
            );
            new_c->population.clear();

            int take = std::min(target_size, total_inds - ind_offset);
            for (int i = 0; i < take; ++i) {
                new_c->population.push_back(all_individuals[ind_offset + i]);
            }
            ind_offset += take;

            while (new_c->size() < target_size) {
                new_c->population.push_back(new_c->make_random_individual());
            }

            new_c->update_individual_type();
            new_c->sort_population();
            countries_.push_back(std::move(new_c));
        }

        for (int i = ind_offset; i < total_inds; ++i) {
            add_individual_to_random_country(all_individuals[i]);
        }
    }
};

// using CountriesAlgorithmMethod = CountriesAlgorithm;