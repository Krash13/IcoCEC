// countries_algorithm_v2.hpp
// Standalone C++ implementation of the Countries Algorithm (ICO), with no
// pybind11/Python dependency. The public start() interface is compatible with
// main_cec2017.cpp and benchmark.hpp.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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

static inline double rand_normal(double mean = 0.0, double stddev = 1.0) {
    return std::normal_distribution<double>{mean, stddev}(rng_engine);
}

static inline int rand_int(int lo, int hi_inclusive) {
    return std::uniform_int_distribution<int>{lo, hi_inclusive}(rng_engine);
}

static inline uint64_t rand_uint64(uint64_t lo, uint64_t hi_inclusive) {
    return std::uniform_int_distribution<uint64_t>{lo, hi_inclusive}(rng_engine);
}

// ============================================================================
// Weighted country-action selection
// 0: Motion, 1: Trade, 2: War, 3: Epidemic, 4: Migration
// ============================================================================

static inline int weighted_action_choice(double p_motion, double p_trade, double p_war, double p_epidemic, double p_migration) {
    // The array order intentionally matches the integer action codes above.
    double probs[5] = {p_motion, p_trade, p_war, p_epidemic, p_migration};
    double sum = probs[0] + probs[1] + probs[2] + probs[3] + probs[4];
    // A zero-sum configuration falls back to an unbiased action instead of
    // leaving the country idle forever.
    if (sum <= 0.0) return rand_int(0, 4);

    double r = rand_uniform(0.0, sum);
    double acc = 0.0;
    for (int i = 0; i < 5; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return 4;
}

// Adaptive probability matching for two competing reproduction operators.
// A failed operator receives zero credit, so its EMA reward decays and its
// probability is automatically redistributed to the competitor.
struct AdaptiveOperatorPair {
    std::array<double, 2> reward = {1e-6, 1e-6};
    std::array<double, 2> probs  = {0.5, 0.5};

    void update(int idx, double credit, double alpha) {
        if (idx < 0 || idx >= 2) return;
        alpha = std::clamp(alpha, 0.0, 1.0);
        credit = std::max(0.0, credit);
        reward[idx] = (1.0 - alpha) * reward[idx] + alpha * credit;
    }

    void renormalize(double floor0, double floor1) {
        floor0 = std::max(0.0, floor0);
        floor1 = std::max(0.0, floor1);
        const double floor_sum = floor0 + floor1;
        if (floor_sum >= 1.0) {
            probs = {floor0 / floor_sum, floor1 / floor_sum};
            return;
        }
        const double sum = reward[0] + reward[1];
        const double residual = 1.0 - floor_sum;
        if (sum <= 1e-30) {
            probs = {floor0 + 0.5 * residual, floor1 + 0.5 * residual};
            return;
        }
        probs[0] = floor0 + residual * reward[0] / sum;
        probs[1] = floor1 + residual * reward[1] / sum;
    }
};

static inline double normalized_elite_credit(double elite_threshold, double child_f) {
    if (!std::isfinite(elite_threshold) || !std::isfinite(child_f) || child_f >= elite_threshold) {
        return 0.0;
    }
    const double gain = elite_threshold - child_f;
    const double scale = std::abs(elite_threshold) + std::abs(child_f) + 1.0;
    return gain / scale;
}

// ============================================================================
// Individual Types & Base Individual
// ============================================================================

enum class IndividualType { Gray, Real };

using FuncT = std::function<double(const std::vector<double>&)>;

struct Individual {
    std::vector<double> x_min, x_max;
    double f_value = std::numeric_limits<double>::infinity();
    // Number of epidemic mutations already applied to this individual. It
    // reduces mutation amplitude and the Gray-code bit-flip count over time.
    int n_ep = 0;
    IndividualType itype;

    Individual(std::vector<double> x_min, std::vector<double> x_max, IndividualType t)
        : x_min(std::move(x_min)), x_max(std::move(x_max)), itype(t) {}

    virtual ~Individual() = default;
    virtual std::vector<double> real_x() const = 0;
    virtual std::shared_ptr<Individual> clone() const = 0;

    bool operator<(const Individual& o) const noexcept { return f_value < o.f_value; }
    bool operator>(const Individual& o) const noexcept { return f_value > o.f_value; }
    bool operator<=(const Individual& o) const noexcept { return f_value <= o.f_value; }
};

// ============================================================================
// Gray Individual
// ============================================================================

struct GrayIndividual : Individual {
    std::vector<int> genes;
    std::vector<uint64_t> code;
    std::vector<double> steps;

    GrayIndividual(std::vector<uint64_t> gray_code,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   const std::vector<int>& genes,
                   const FuncT& func)
        : Individual(x_min, x_max, IndividualType::Gray),
          genes(genes),
          code(std::move(gray_code)) {
        init_steps();
        f_value = func(real_x());
    }

    GrayIndividual(std::vector<uint64_t> gray_code,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   const std::vector<int>& genes,
                   double cached_f_value)
        : Individual(x_min, x_max, IndividualType::Gray),
          genes(genes),
          code(std::move(gray_code)) {
        init_steps();
        f_value = cached_f_value;
    }

    std::vector<double> real_x() const override {
        // Gray code is decoded to an integer grid and then mapped linearly to
        // the bounded real-valued search space.
        int dim = (int)genes.size();
        std::vector<double> rx(dim);
        for (int i = 0; i < dim; ++i) {
            rx[i] = x_min[i] + steps[i] * (double)gray_code_to_tc(code[i]);
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
        const std::vector<double>& x_min,
        const std::vector<double>& x_max,
        const std::vector<int>& genes,
        const FuncT& func) {
        int dim = (int)genes.size();
        std::vector<uint64_t> gc(dim);
        for (int i = 0; i < dim; ++i) {
            uint64_t max_val = (1ULL << genes[i]) - 1;
            gc[i] = tc_to_gray_code(std::min(decimal[i], max_val));
        }
        return std::make_shared<GrayIndividual>(gc, x_min, x_max, genes, func);
    }

    static std::shared_ptr<GrayIndividual> from_real(
        const std::vector<double>& x,
        const std::vector<double>& x_min,
        const std::vector<double>& x_max,
        const std::vector<int>& genes,
        const FuncT& func) {

        const int dim = static_cast<int>(genes.size());
        std::vector<uint64_t> gray_code(dim);

        for (int i = 0; i < dim; ++i) {
            const uint64_t max_val = (1ULL << genes[i]) - 1ULL;
            const double range = x_max[i] - x_min[i];

            double normalized = 0.0;
            if (range > 0.0) {
                normalized =
                    (std::clamp(x[i], x_min[i], x_max[i]) - x_min[i]) / range;
            }

            normalized = std::clamp(normalized, 0.0, 1.0);

            const uint64_t decimal = static_cast<uint64_t>(
                std::llround(static_cast<long double>(normalized) *
                             static_cast<long double>(max_val))
            );

            gray_code[i] = tc_to_gray_code(std::min(decimal, max_val));
        }

        return std::make_shared<GrayIndividual>(
            std::move(gray_code), x_min, x_max, genes, func
        );
    }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
        p->n_ep = n_ep;
        return p;
    }

    std::shared_ptr<GrayIndividual> mutate(double q_max_term, const FuncT& func) const {
        // ICO mutation schedule (formula 7):
        // q_c = max(0, floor((1 - t / tmax) * q_max - n_ep)).
        int n = std::max(0, (int)std::floor(q_max_term - n_ep));
        if (n == 0) {
            // Reuse the cached objective value because the genotype is
            // unchanged and no new function evaluation is required.
            auto p = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
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
        // Partial Fisher-Yates shuffle selects distinct bits to flip.
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

        auto p = std::make_shared<GrayIndividual>(new_code, x_min, x_max, genes, func);
        return p;
    }

    // Uniform and two-point crossover from the original binary implementation.
    // The operator itself is selected in Country::reproduction, so the old
    // hard-coded 0.7 probability is replaced with an explicit choice.
    static std::pair<std::shared_ptr<GrayIndividual>, std::shared_ptr<GrayIndividual>>
    crossover(const GrayIndividual& a, const GrayIndividual& b, bool uniform, const FuncT& func) {
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

        if (uniform) {
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
            return std::make_shared<GrayIndividual>(gc, a.x_min, a.x_max, a.genes, func);
        };

        return {bits_to_ind(nb1), bits_to_ind(nb2)};
    }

    // Gray code is only a representation and quantizer for Eigen crossover.
    // The arithmetic operator works in the covariance eigenvector coordinate system.
    static std::pair<
        std::shared_ptr<GrayIndividual>,
        std::shared_ptr<GrayIndividual>>
    eigen_crossover(
        const GrayIndividual& a,
        const GrayIndividual& b,
        const Eigen::MatrixXd* eigen_basis,
        const FuncT& func) {

        const int dim = static_cast<int>(a.genes.size());
        assert(dim == static_cast<int>(b.genes.size()));

        const auto ax_real = a.real_x();
        const auto bx_real = b.real_x();

        // Work in normalized [0, 1]^D coordinates so dimensions with larger
        // physical ranges do not dominate the covariance matrix.
        Eigen::VectorXd ax(dim);
        Eigen::VectorXd bx(dim);

        for (int d = 0; d < dim; ++d) {
            const double range = a.x_max[d] - a.x_min[d];

            if (range > 0.0) {
                ax[d] = (ax_real[d] - a.x_min[d]) / range;
                bx[d] = (bx_real[d] - a.x_min[d]) / range;
            } else {
                ax[d] = 0.0;
                bx[d] = 0.0;
            }
        }

        const bool use_eigen =
            eigen_basis != nullptr &&
            eigen_basis->rows() == dim &&
            eigen_basis->cols() == dim &&
            eigen_basis->allFinite();

        Eigen::VectorXd za;
        Eigen::VectorXd zb;

        if (use_eigen) {
            // x' = B^T x
            za = eigen_basis->transpose() * ax;
            zb = eigen_basis->transpose() * bx;
        } else {
            za = ax;
            zb = bx;
        }

        Eigen::VectorXd z1 = za;
        Eigen::VectorXd z2 = zb;

        // Arithmetic crossover uses an independent alpha for every Eigen
        // direction. A single alpha for the whole vector would cancel the
        // rotation and reduce to an ordinary interpolation between parents.
        for (int j = 0; j < dim; ++j) {
            const double alpha = rand_uniform(0.0, 1.0);
            z1[j] = za[j] + alpha * (zb[j] - za[j]);
            z2[j] = zb[j] + alpha * (za[j] - zb[j]);
        }

        // Previous binomial Eigen crossover is kept commented for A/B experiments.
        // Helper:
        // auto binomial_eigen_crossover = [&](Eigen::VectorXd& child_z,
        //                                      const Eigen::VectorXd& other_z,
        //                                      double cr) {
        //     const int j_rand = rand_int(0, dim - 1);
        //     for (int j = 0; j < dim; ++j) {
        //         if (j == j_rand || rand_uniform(0.0, 1.0) <= cr) {
        //             child_z[j] = other_z[j];
        //         }
        //     }
        // };
        //
        // Calls (instead of the arithmetic loop above):
        // const double gray_eigen_cr = 0.80;
        // z1 = za;
        // z2 = zb;
        // binomial_eigen_crossover(z1, zb, gray_eigen_cr);
        // binomial_eigen_crossover(z2, za, gray_eigen_cr);

        Eigen::VectorXd child1;
        Eigen::VectorXd child2;

        if (use_eigen) {
            // x = B x'
            child1 = (*eigen_basis) * z1;
            child2 = (*eigen_basis) * z2;
        } else {
            child1 = std::move(z1);
            child2 = std::move(z2);
        }

        // Eigen crossover can leave the normalized hyper-box. Repair toward
        // the corresponding parent midpoint rather than clipping hard.
        auto repair = [](Eigen::VectorXd& x, const Eigen::VectorXd& parent) {
            for (Eigen::Index d = 0; d < x.size(); ++d) {
                if (x[d] < 0.0) {
                    x[d] = 0.5 * parent[d];
                } else if (x[d] > 1.0) {
                    x[d] = 0.5 * (parent[d] + 1.0);
                }
            }
        };

        repair(child1, ax);
        repair(child2, bx);

        std::vector<double> x1(dim);
        std::vector<double> x2(dim);

        for (int d = 0; d < dim; ++d) {
            const double range = a.x_max[d] - a.x_min[d];
            x1[d] = a.x_min[d] + std::clamp(child1[d], 0.0, 1.0) * range;
            x2[d] = a.x_min[d] + std::clamp(child2[d], 0.0, 1.0) * range;
        }

        return {
            from_real(x1, a.x_min, a.x_max, a.genes, func),
            from_real(x2, a.x_min, a.x_max, a.genes, func)
        };
    }

private:
    void init_steps() {
        int dim = (int)genes.size();
        steps.resize(dim);
        for (int i = 0; i < dim; ++i) {
            steps[i] = (x_max[i] - x_min[i]) / (double)((1ULL << genes[i]) - 1);
        }
    }
};

// ============================================================================
// Real Individual
// ============================================================================

struct RealIndividual : Individual {
    std::vector<double> x;

    RealIndividual(std::vector<double> x,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   const FuncT& func)
        : Individual(x_min, x_max, IndividualType::Real), x(std::move(x)) {
        f_value = func(this->x);
    }

    RealIndividual(std::vector<double> x,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   double cached_f_value)
        : Individual(x_min, x_max, IndividualType::Real), x(std::move(x)) {
        f_value = cached_f_value;
    }

    std::vector<double> real_x() const override { return x; }

    std::shared_ptr<Individual> clone() const override {
        auto p = std::make_shared<RealIndividual>(x, x_min, x_max, f_value);
        p->n_ep = n_ep;
        return p;
    }

    void update_f(const FuncT& func) {
        f_value = func(x);
    }

    static std::shared_ptr<RealIndividual> differential_crossover(
        const RealIndividual& target,
        const RealIndividual& pbest,
        const RealIndividual& r1,
        const RealIndividual& r2,
        double F,
        double CR,
        const std::vector<double>& x_min,
        const std::vector<double>& x_max,
        const FuncT& func,
        double* actual_cr = nullptr) {

        const int dim = static_cast<int>(target.x.size());
        std::vector<double> child_x = target.x;
        const int forced_dim = rand_int(0, dim - 1);
        int crossed = 0;

        for (int d = 0; d < dim; ++d) {
            if (rand_uniform(0.0, 1.0) < CR || d == forced_dim) {
                double value =
                    target.x[d] +
                    F * (pbest.x[d] - target.x[d]) +
                    F * (r1.x[d] - r2.x[d]);

                // Match the competition DE implementations: an infeasible
                // donor component is resampled inside the legal interval.
                if (value < x_min[d] || value > x_max[d]) {
                    value = rand_uniform(x_min[d], x_max[d]);
                }
                child_x[d] = value;
                crossed++;
            }
        }

        if (actual_cr != nullptr) {
            *actual_cr = static_cast<double>(crossed) / std::max(1, dim);
        }

        return std::make_shared<RealIndividual>(
            std::move(child_x), x_min, x_max, func);
    }

    void mutation(const FuncT& func, double p_max) {
        n_ep += 1;
        int dim = (int)x.size();
        for (int i = 0; i < dim; ++i) {
            double range = x_max[i] - x_min[i];
            // Scale by the coordinate range, then decay the step as the same
            // individual survives additional epidemics.
            double perturb = p_max * rand_uniform(-0.5, 0.5) * range / (double)n_ep;
            x[i] = std::clamp(x[i] + perturb, x_min[i], x_max[i]);
        }
        update_f(func);
    }

    static std::shared_ptr<RealIndividual> crossover(
        const RealIndividual& a, const RealIndividual& b,
        double alpha,
        const std::vector<double>& x_min,
        const std::vector<double>& x_max,
        const FuncT& func) {
        const Eigen::Index dim = static_cast<Eigen::Index>(a.x.size());
        Eigen::Map<const Eigen::ArrayXd> ax(a.x.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> bx(b.x.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> min_x(x_min.data(), dim);
        Eigen::Map<const Eigen::ArrayXd> max_x(x_max.data(), dim);

        const Eigen::ArrayXd lo = ax.min(bx);
        const Eigen::ArrayXd hi = ax.max(bx);
        const Eigen::ArrayXd interval = hi - lo;
        const Eigen::ArrayXd lower = lo - alpha * interval;
        const Eigen::ArrayXd upper = hi + alpha * interval;

        Eigen::ArrayXd u(dim);
        for (Eigen::Index i = 0; i < dim; ++i) {
            u(i) = rand_uniform(0.0, 1.0);
        }

        // BLX-alpha crossover samples the expanded parental interval and then
        // clamps all coordinates to the legal search bounds.
        Eigen::ArrayXd child = lower + u * (upper - lower);
        child = child.max(min_x).min(max_x);

        std::vector<double> new_x(child.data(), child.data() + child.size());
        return std::make_shared<RealIndividual>(std::move(new_x), x_min, x_max, func);
    }

    static std::shared_ptr<RealIndividual> eigen_crossover(
        const RealIndividual& a, const RealIndividual& b,
        const Eigen::MatrixXd* eigen_basis,
        const std::vector<double>& x_min,
        const std::vector<double>& x_max,
        const FuncT& func) {
        const int dim = static_cast<int>(a.x.size());
        assert(dim == static_cast<int>(b.x.size()));

        Eigen::VectorXd ax(dim);
        Eigen::VectorXd bx(dim);

        // Use the same normalized coordinate system as the covariance matrix.
        for (int d = 0; d < dim; ++d) {
            const double range = x_max[d] - x_min[d];
            if (range > 0.0) {
                ax[d] = (a.x[d] - x_min[d]) / range;
                bx[d] = (b.x[d] - x_min[d]) / range;
            } else {
                ax[d] = 0.0;
                bx[d] = 0.0;
            }
        }

        const bool use_eigen =
            eigen_basis != nullptr &&
            eigen_basis->rows() == dim &&
            eigen_basis->cols() == dim &&
            eigen_basis->allFinite();

        Eigen::VectorXd za;
        Eigen::VectorXd zb;

        if (use_eigen) {
            za = eigen_basis->transpose() * ax;
            zb = eigen_basis->transpose() * bx;
        } else {
            za = ax;
            zb = bx;
        }

        Eigen::VectorXd child_z = za;

        // Arithmetic crossover along independently selected Eigen directions.
        for (int j = 0; j < dim; ++j) {
            const double alpha = rand_uniform(0.0, 1.0);
            child_z[j] = za[j] + alpha * (zb[j] - za[j]);
        }

        // Previous binomial Eigen crossover is kept commented for A/B experiments.
        // Helper:
        // auto binomial_eigen_crossover = [&](Eigen::VectorXd& result_z,
        //                                      const Eigen::VectorXd& other_z,
        //                                      double cr) {
        //     const int j_rand = rand_int(0, dim - 1);
        //     for (int j = 0; j < dim; ++j) {
        //         if (j == j_rand || rand_uniform(0.0, 1.0) <= cr) {
        //             result_z[j] = other_z[j];
        //         }
        //     }
        // };
        //
        // Call (instead of the arithmetic loop above):
        // const double real_eigen_cr = 0.80;
        // child_z = za;
        // binomial_eigen_crossover(child_z, zb, real_eigen_cr);

        Eigen::VectorXd child = use_eigen ? (*eigen_basis) * child_z : child_z;

        // Repair toward the first parent instead of hard clipping.
        for (int d = 0; d < dim; ++d) {
            if (child[d] < 0.0) {
                child[d] = 0.5 * ax[d];
            } else if (child[d] > 1.0) {
                child[d] = 0.5 * (ax[d] + 1.0);
            }
        }

        std::vector<double> new_x(dim);
        for (int d = 0; d < dim; ++d) {
            const double range = x_max[d] - x_min[d];
            new_x[d] = x_min[d] + std::clamp(child[d], 0.0, 1.0) * range;
        }

        return std::make_shared<RealIndividual>(std::move(new_x), x_min, x_max, func);
    }
};

// ============================================================================
// Country
// ============================================================================

struct Country {
    std::vector<std::shared_ptr<Individual>> population;
    std::vector<double> x_min, x_max;
    std::vector<int> genes;
    FuncT f;
    int N;
    IndividualType itype;

    int action = -1;
    Country* ally = nullptr;
    Country* enemy = nullptr;

    // Success-history state for the Real DE branch. Kept per country because
    // each country can occupy a different basin and therefore prefer a
    // different mutation/crossover scale.
    std::vector<double> de_memory_f;
    std::vector<double> de_memory_cr;
    int de_memory_index = 0;
    double de_success_rate = 0.5;

    Country(int N,
            const std::vector<double>& x_min,
            const std::vector<double>& x_max,
            const FuncT& func,
            IndividualType it,
            const std::vector<int>& genes)
        : x_min(x_min), x_max(x_max), genes(genes), f(func), N(N), itype(it) {
        int dim = (int)x_min.size();
        population.reserve(N);

        if (itype == IndividualType::Gray) {
            // Each country starts inside a randomly selected hyper-rectangle,
            // expressed on the integer grid used by the Gray representation.
            std::vector<uint64_t> local_min(dim), local_max(dim);
            for (int d = 0; d < dim; ++d) {
                uint64_t max_val = (1ULL << genes[d]) - 1;
                local_min[d] = rand_uint64(0, max_val - 1);
                local_max[d] = rand_uint64(local_min[d] + 1, max_val);
            }
            for (int i = 0; i < N; ++i) {
                std::vector<uint64_t> dec(dim);
                for (int d = 0; d < dim; ++d) {
                    dec[d] = rand_uint64(local_min[d], local_max[d]);
                }
                population.push_back(GrayIndividual::from_decimal(dec, x_min, x_max, genes, f));
            }
        } else {
            // Real-coded countries use the same localized initialization idea
            // directly in the continuous coordinate space.
            std::vector<double> local_min(dim), local_max(dim);
            for (int d = 0; d < dim; ++d) {
                local_min[d] = rand_uniform(x_min[d], x_max[d]);
                local_max[d] = rand_uniform(local_min[d], x_max[d]);
            }
            for (int i = 0; i < N; ++i) {
                std::vector<double> x(dim);
                for (int d = 0; d < dim; ++d) {
                    x[d] = rand_uniform(local_min[d], local_max[d]);
                }
                population.push_back(std::make_shared<RealIndividual>(std::move(x), x_min, x_max, f));
            }
        }
        sort_population();
    }

    bool empty() const noexcept { return population.empty(); }
    int size() const noexcept { return (int)population.size(); }

    void sort_population() {
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a->f_value < b->f_value; });
    }

    double best_f() const noexcept {
        return population.empty() ? std::numeric_limits<double>::infinity() : population[0]->f_value;
    }

    double avg_f() const noexcept {
        if (population.empty()) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (const auto& ind : population) s += ind->f_value;
        return s / (double)population.size();
    }

    int select_rank_biased_parent(int parent_count, double pressure, int exclude = -1) const {
        if (parent_count <= 1) return 0;

        pressure = std::max(0.0, pressure);
        std::vector<double> weights(parent_count, 0.0);
        const double denom = static_cast<double>(std::max(1, parent_count - 1));

        // population is sorted best-to-worst. Exponential rank weights are
        // invariant to objective shifts/scales, unlike raw roulette fitness.
        for (int rank = 0; rank < parent_count; ++rank) {
            if (rank == exclude) continue;
            weights[rank] = std::exp(-pressure * static_cast<double>(rank) / denom);
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return dist(rng_engine);
    }

    void select_action(std::vector<Country*>& all_countries,
                       double p_motion, double p_trade, double p_war, double p_epidemic, double p_migration) {
        action = weighted_action_choice(p_motion, p_trade, p_war, p_epidemic, p_migration);

        if (action == 1) {
            // Trade and war are paired actions. Reserve an unassigned partner
            // now so that it cannot be selected by another country this round.
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
                // Pairing is impossible for the last unassigned country.
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

    void do_motion(double r_max = 2.0) {
        if (population.empty()) { action = -1; return; }

        if (itype == IndividualType::Gray) {
            auto best_dec = std::static_pointer_cast<GrayIndividual>(population[0])->decimal_x();
            for (int i = 1; i < size(); ++i) {
                auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
                auto dec = ind->decimal_x();
                double r = rand_uniform(0.0, r_max);
                std::vector<uint64_t> new_dec(genes.size());
                for (int d = 0; d < (int)genes.size(); ++d) {
                    int64_t diff = static_cast<int64_t>(best_dec[d]) - static_cast<int64_t>(dec[d]);
                    int64_t nd = static_cast<int64_t>(dec[d]) + static_cast<int64_t>(r * static_cast<double>(diff));
                    uint64_t max_val = (1ULL << genes[d]) - 1;
                    new_dec[d] = static_cast<uint64_t>(std::clamp(nd, (int64_t)0, (int64_t)max_val));
                }
                population[i] = GrayIndividual::from_decimal(new_dec, x_min, x_max, genes, f);
            }
        } else {
            const auto& best_x = std::static_pointer_cast<RealIndividual>(population[0])->x;
            for (int i = 1; i < size(); ++i) {
                auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
                // One scalar is sampled per individual, so motion remains on
                // the current-to-leader line (important for rotated problems).
                double r = rand_uniform(0.0, r_max);
                for (int d = 0; d < (int)ind->x.size(); ++d) {
                    ind->x[d] = std::clamp(ind->x[d] + r * (best_x[d] - ind->x[d]), x_min[d], x_max[d]);
                }
                ind->update_f(f);
            }
        }
        sort_population();
        action = -1;
    }

    void do_epidemic(double elite_frac, double dead_frac, double p_max_real, double q_max_term_gray) {
        int n = size();
        int n_elite = (int)std::ceil(elite_frac * n);
        int n_dead  = (int)std::ceil(dead_frac * n);

        if (n_dead >= n) {
            population.clear();
            action = -1;
            return;
        }

        // The population is sorted, so removing from the tail kills the worst
        // individuals while the first n_elite individuals remain untouched.
        if (n_dead > 0) {
            population.erase(population.end() - n_dead, population.end());
        }

        for (int i = n_elite; i < size(); ++i) {
            if (itype == IndividualType::Gray) {
                auto ind = std::static_pointer_cast<GrayIndividual>(population[i]);
                auto new_ind = ind->mutate(q_max_term_gray, f);
                new_ind->n_ep = ind->n_ep + 1;
                population[i] = new_ind;
            } else {
                auto ind = std::static_pointer_cast<RealIndividual>(population[i]);
                ind->mutation(f, p_max_real);
            }
        }
        sort_population();
        action = -1;
    }

    void do_migration(double migrate_frac = 0.3) {
        int n = size();
        if (n <= 1) { action = -1; return; }

        int n_migrate = std::clamp((int)std::ceil(migrate_frac * n), 1, n - 1);
        // Replace the worst fraction with globally sampled individuals to
        // inject diversity without discarding the country's leader.
        population.erase(population.end() - n_migrate, population.end());

        for (int i = 0; i < n_migrate; ++i) {
            population.push_back(make_random_individual());
        }
        sort_population();
        action = -1;
    }

    void reproduction(
        int n_min, int n_max,
        double p_min, double p_max,
        double f_min, double f_max,
        int iteration, int t_max,
        const Eigen::MatrixXd* eigen_basis,
        double real_blx_share,
        double real_eigen_share,
        double real_de_share,
        double de_f,
        double de_cr,
        double de_pbest_frac,
        double de_pool_frac,
        bool de_adaptive,
        int de_memory_size,
        double de_f_sigma,
        double de_cr_sigma,
        double gray_uniform_share,
        double gray_two_point_share,
        double gray_eigen_share,
        double parent_rank_pressure,
        AdaptiveOperatorPair* real_operator_adapt,
        AdaptiveOperatorPair* gray_operator_adapt,
        double operator_alpha) {

        if (size() < 2) return;

        const double avg = avg_f();

        // Better countries reproduce more. The small denominator guard also
        // keeps the formula defined when country averages coincide.
        const double n_frac = (f_max - avg) / (f_max - f_min + 1e-15);
        const int n = std::clamp(
            static_cast<int>(std::ceil((n_max - n_min) * n_frac + n_min)),
            n_min,
            n_max
        );

        // Children produced earlier in this call must not become parents in
        // the same generation.
        const int parent_count = size();

        std::vector<double> successful_f;
        std::vector<double> successful_cr;
        std::vector<double> successful_delta;
        int de_attempts = 0;
        int de_successes = 0;

        if (itype == IndividualType::Gray) {
            if (gray_operator_adapt != nullptr) {
                gray_uniform_share = gray_operator_adapt->probs[0];
                gray_two_point_share = gray_operator_adapt->probs[1];
                gray_eigen_share = 0.0;
            }
            gray_uniform_share = std::max(0.0, gray_uniform_share);
            gray_two_point_share = std::max(0.0, gray_two_point_share);
            gray_eigen_share = std::max(0.0, gray_eigen_share);
            double crossover_share_sum = gray_uniform_share + gray_two_point_share + gray_eigen_share;
            const int gray_elite_count = std::clamp(
                static_cast<int>(std::ceil(0.20 * parent_count)), 1, parent_count);
            const double gray_elite_threshold = population[gray_elite_count - 1]->f_value;

            if (crossover_share_sum <= 0.0) {
                gray_uniform_share = 0.05;
                gray_two_point_share = 0.05;
                gray_eigen_share = 0.90;
                crossover_share_sum = 1.0;
            }

            for (int i = 0; i < n; ++i) {
                const int k1 = select_rank_biased_parent(
                    parent_count, parent_rank_pressure);
                const int k2 = select_rank_biased_parent(
                    parent_count, parent_rank_pressure, k1);

                auto a = std::static_pointer_cast<GrayIndividual>(population[k1]);
                auto b = std::static_pointer_cast<GrayIndividual>(population[k2]);

                const double crossover_choice = rand_uniform(0.0, crossover_share_sum);
                std::pair<std::shared_ptr<GrayIndividual>, std::shared_ptr<GrayIndividual>> children;

                int operator_idx = 0;
                if (crossover_choice < gray_uniform_share) {
                    children = GrayIndividual::crossover(*a, *b, true, f);
                    operator_idx = 0;
                } else if (crossover_choice < gray_uniform_share + gray_two_point_share) {
                    children = GrayIndividual::crossover(*a, *b, false, f);
                    operator_idx = 1;
                } else {
                    // Eigen crossover is disabled in the clean DE branch.
                    children = GrayIndividual::crossover(*a, *b, false, f);
                    operator_idx = 1;
                }

                if (gray_operator_adapt != nullptr) {
                    gray_operator_adapt->update(
                        operator_idx,
                        normalized_elite_credit(gray_elite_threshold, children.first->f_value),
                        operator_alpha);
                    gray_operator_adapt->update(
                        operator_idx,
                        normalized_elite_credit(gray_elite_threshold, children.second->f_value),
                        operator_alpha);
                }

                population.push_back(std::move(children.first));
                population.push_back(std::move(children.second));
            }
        } else {
            const double p = std::clamp(
                p_max - (p_max - p_min) *
                    (1.0 - static_cast<double>(iteration) / t_max) *
                    ((avg - f_min) / (f_max - f_min + 1e-15)),
                p_min, p_max
            );

            if (real_operator_adapt != nullptr) {
                real_blx_share = real_operator_adapt->probs[0];
                real_eigen_share = 0.0;
                real_de_share = real_operator_adapt->probs[1];
            }
            real_blx_share = std::max(0.0, real_blx_share);
            real_eigen_share = std::max(0.0, real_eigen_share);
            real_de_share = std::max(0.0, real_de_share);
            double crossover_share_sum = real_blx_share + real_eigen_share + real_de_share;

            if (crossover_share_sum <= 0.0) {
                real_blx_share = 0.35;
                real_eigen_share = 0.35;
                real_de_share = 0.30;
                crossover_share_sum = 1.0;
            }

            de_f = std::clamp(de_f, 0.0, 1.0);
            de_cr = std::clamp(de_cr, 0.0, 1.0);
            de_pbest_frac = std::clamp(de_pbest_frac, 0.0, 1.0);
            de_pool_frac = std::clamp(de_pool_frac, 0.0, 1.0);
            de_memory_size = std::max(1, de_memory_size);

            if (de_memory_f.size() != static_cast<size_t>(de_memory_size)) {
                de_memory_f.assign(de_memory_size, de_f);
                de_memory_cr.assign(de_memory_size, de_cr);
                de_memory_index = 0;
                de_success_rate = 0.5;
            }

            const int de_pool_count = std::clamp(
                static_cast<int>(std::ceil(de_pool_frac * parent_count)),
                std::min(4, parent_count), parent_count
            );
            const int de_elite_count = std::clamp(
                static_cast<int>(std::ceil(de_pbest_frac * de_pool_count)),
                std::min(2, de_pool_count), de_pool_count
            );
            const double de_elite_threshold = population[de_elite_count - 1]->f_value;

            for (int i = 0; i < 2 * n; ++i) {
                const double crossover_choice = rand_uniform(0.0, crossover_share_sum);

                if (crossover_choice >= real_blx_share + real_eigen_share && parent_count >= 4) {
                    // L-SRTDE/RDEx-style current-to-pbest/1. The target and
                    // second differential donor remain uniform, while r1 is
                    // rank-biased and pbest comes from the elite prefix.
                    // Keep DE inside the best prefix of the country. Very bad
                    // outliers stay available to diversification operators but
                    // cannot distort the differential geometry of exploitation.
                    const int target_idx = rand_int(0, de_pool_count - 1);
                    const int pbest_count = de_elite_count;

                    int pbest_idx = rand_int(0, pbest_count - 1);
                    while (pbest_idx == target_idx) {
                        pbest_idx = rand_int(0, pbest_count - 1);
                    }

                    int r1_idx = select_rank_biased_parent(
                        de_pool_count, parent_rank_pressure, target_idx);
                    while (r1_idx == pbest_idx) {
                        r1_idx = select_rank_biased_parent(
                            de_pool_count, parent_rank_pressure, target_idx);
                    }

                    int r2_idx = rand_int(0, de_pool_count - 1);
                    while (r2_idx == target_idx || r2_idx == pbest_idx || r2_idx == r1_idx) {
                        r2_idx = rand_int(0, parent_count - 1);
                    }

                    auto target = std::static_pointer_cast<RealIndividual>(population[target_idx]);
                    auto pbest = std::static_pointer_cast<RealIndividual>(population[pbest_idx]);
                    auto r1 = std::static_pointer_cast<RealIndividual>(population[r1_idx]);
                    auto r2 = std::static_pointer_cast<RealIndividual>(population[r2_idx]);

                    double sampled_f = de_f;
                    double sampled_cr = de_cr;
                    if (de_adaptive) {
                        const int memory_slot = rand_int(0, de_memory_size - 1);

                        // SHADE/RDEx-style success-history sampling: F uses a
                        // heavy-tailed Cauchy perturbation so occasional larger
                        // differential steps remain possible; CR uses Gaussian.
                        std::cauchy_distribution<double> f_dist(
                            de_memory_f[memory_slot], std::max(1e-12, de_f_sigma));
                        do {
                            sampled_f = f_dist(rng_engine);
                        } while (sampled_f <= 0.0);
                        sampled_f = std::min(sampled_f, 1.0);

                        sampled_cr = std::normal_distribution<double>{
                            de_memory_cr[memory_slot], std::max(1e-12, de_cr_sigma)
                        }(rng_engine);
                        sampled_cr = std::clamp(sampled_cr, 0.0, 1.0);
                    }

                    double actual_cr = sampled_cr;
                    auto child = RealIndividual::differential_crossover(
                        *target, *pbest, *r1, *r2,
                        sampled_f, sampled_cr, x_min, x_max, f, &actual_cr);

                    de_attempts++;
                    const double improvement = target->f_value - child->f_value;
                    // Success-history is learned only from offspring that are
                    // actually competitive with the current elite region. A
                    // huge improvement of a catastrophic target (for example
                    // 1e10 -> 1e9 while elite fitness is ~1e3) must not dominate
                    // F/CR memory updates.
                    if (improvement > 0.0 && child->f_value <= de_elite_threshold) {
                        de_successes++;
                        successful_f.push_back(sampled_f);
                        successful_cr.push_back(actual_cr);
                        const double scale = std::abs(target->f_value) +
                                             std::abs(child->f_value) + 1.0;
                        successful_delta.push_back(improvement / scale);
                    }
                    if (real_operator_adapt != nullptr) {
                        real_operator_adapt->update(
                            1,
                            normalized_elite_credit(de_elite_threshold, child->f_value),
                            operator_alpha);
                    }

                    population.push_back(std::move(child));
                    continue;
                }

                const int k1 = select_rank_biased_parent(
                    parent_count, parent_rank_pressure);
                const int k2 = select_rank_biased_parent(
                    parent_count, parent_rank_pressure, k1);

                auto a = std::static_pointer_cast<RealIndividual>(population[k1]);
                auto b = std::static_pointer_cast<RealIndividual>(population[k2]);

                if (crossover_choice < real_blx_share || parent_count < 4) {
                    auto child = RealIndividual::crossover(*a, *b, p, x_min, x_max, f);
                    if (real_operator_adapt != nullptr) {
                        real_operator_adapt->update(
                            0,
                            normalized_elite_credit(de_elite_threshold, child->f_value),
                            operator_alpha);
                    }
                    population.push_back(std::move(child));
                } else {
                    // Eigen crossover is disabled; any numerical fall-through
                    // is repaired to BLX so covariance never re-enters the run.
                    auto child = RealIndividual::crossover(*a, *b, p, x_min, x_max, f);
                    if (real_operator_adapt != nullptr) {
                        real_operator_adapt->update(
                            0,
                            normalized_elite_credit(de_elite_threshold, child->f_value),
                            operator_alpha);
                    }
                    population.push_back(std::move(child));
                }
            }
        }

        if (itype == IndividualType::Real && de_attempts > 0) {
            de_success_rate = static_cast<double>(de_successes) /
                              static_cast<double>(de_attempts);

            if (de_adaptive && !successful_f.empty()) {
                double delta_sum = 0.0;
                for (double delta : successful_delta) delta_sum += delta;

                double f_num = 0.0, f_den = 0.0;
                double cr_num = 0.0, cr_den = 0.0;
                for (size_t i = 0; i < successful_f.size(); ++i) {
                    const double w = (delta_sum > 1e-30)
                        ? successful_delta[i] / delta_sum
                        : 1.0 / static_cast<double>(successful_f.size());

                    f_num += w * successful_f[i] * successful_f[i];
                    f_den += w * successful_f[i];
                    cr_num += w * successful_cr[i] * successful_cr[i];
                    cr_den += w * successful_cr[i];
                }

                if (f_den > 1e-30) {
                    de_memory_f[de_memory_index] = std::clamp(f_num / f_den, 0.0, 1.0);
                }
                if (cr_den > 1e-30) {
                    const double lehmer_cr = std::clamp(cr_num / cr_den, 0.0, 1.0);
                    de_memory_cr[de_memory_index] =
                        0.5 * (de_memory_cr[de_memory_index] + lehmer_cr);
                }

                de_memory_index = (de_memory_index + 1) % de_memory_size;
            }
        }

        sort_population();
    }

    void extinction(int m_min, int m_max, double f_min, double f_max,
                    int min_survivors = 0) {
        double avg = avg_f();
        // Worse countries lose more individuals, complementing the adaptive
        // reproduction rule above. During explicit population reduction, keep
        // a protected core so extinction cannot silently collapse a country
        // below the requested late-stage population floor.
        int m = std::clamp(
            (int)((m_max - m_min) * ((avg - f_min) / (f_max - f_min + 1e-15)) + m_min),
            m_min, m_max
        );

        min_survivors = std::max(0, min_survivors);
        const int removable = std::max(0, size() - min_survivors);
        m = std::min(m, removable);
        if (m <= 0) return;

        population.erase(population.end() - m, population.end());
    }

    void truncate(int max_size) {
        if (size() > max_size) {
            population.resize(max_size);
        }
    }

    void update_individual_type() {
        // A trade or war may move individuals between Gray- and real-coded
        // countries. Convert their representation without re-evaluating the
        // objective because the represented real point is unchanged.
        int dim = (int)x_min.size();
        for (auto& ind : population) {
            if (ind->itype == itype) continue;
            if (itype == IndividualType::Real) {
                auto rx = ind->real_x();
                auto ni = std::make_shared<RealIndividual>(std::move(rx), x_min, x_max, ind->f_value);
                ni->n_ep = ind->n_ep;
                ind = ni;
            } else {
                auto rx = ind->real_x();
                std::vector<uint64_t> gc(dim);
                for (int d = 0; d < dim; ++d) {
                    double step = (x_max[d] - x_min[d]) / (double)((1ULL << genes[d]) - 1);
                    int64_t v = (int64_t)std::round((rx[d] - x_min[d]) / step);
                    uint64_t max_v = (1ULL << genes[d]) - 1;
                    gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)max_v));
                }
                auto ni = std::make_shared<GrayIndividual>(gc, x_min, x_max, genes, ind->f_value);
                ni->n_ep = ind->n_ep;
                ind = ni;
            }
        }
    }

    std::shared_ptr<Individual> make_random_individual() const {
        const int dim = static_cast<int>(x_min.size());
        if (itype == IndividualType::Gray) {
            std::vector<uint64_t> decimal(dim);
            for (int d = 0; d < dim; ++d) {
                const uint64_t max_val = (1ULL << genes[d]) - 1;
                decimal[d] = rand_uint64(0, max_val);
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
        // Small countries exchange at most half of their current population,
        // preventing the operation from emptying either participant.
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
            // Descending erasure preserves the validity of the remaining indices.
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

        // Clones are exchanged and then converted to the representation used
        // by their destination country.
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
        Country& home, double r_max) {
        auto wx = winner.real_x();
        auto lx = loser.real_x();
        const int dim = static_cast<int>(wx.size());

        double dist2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = wx[d] - lx[d];
            dist2 += diff * diff;
        }
        const double dist = std::sqrt(dist2);

        // Generate one isotropic noise direction for the entire individual.
        std::vector<double> noise(dim);
        double nrm2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            noise[d] = rand_uniform(-1.0, 1.0);
            nrm2 += noise[d] * noise[d];
        }
        const double nrm = std::sqrt(nrm2) + 1e-12;

        // Reflect beyond the winner and add a small orthogonal-style jitter;
        // this avoids generating every recruit on a single line.
        const double alpha = rand_uniform(0.1, std::max(0.15, r_max * 0.4));
        const double jitter = 0.05 * dist;

        std::vector<double> nx(dim);
        for (int d = 0; d < dim; ++d) {
            const double reflected = wx[d] + alpha * (wx[d] - lx[d]);
            nx[d] = std::clamp(reflected + jitter * (noise[d] / nrm), home.x_min[d], home.x_max[d]);
        }

        if (home.itype == IndividualType::Real) {
            return std::make_shared<RealIndividual>(std::move(nx), home.x_min, home.x_max, home.f);
        }

        std::vector<uint64_t> dec(dim);
        for (int d = 0; d < dim; ++d) {
            const double step = (home.x_max[d] - home.x_min[d]) / static_cast<double>((1ULL << home.genes[d]) - 1);
            const int64_t v = static_cast<int64_t>(std::round((nx[d] - home.x_min[d]) / step));
            const uint64_t max_v = (1ULL << home.genes[d]) - 1;
            dec[d] = static_cast<uint64_t>(std::clamp(v, (int64_t)0, static_cast<int64_t>(max_v)));
        }
        return GrayIndividual::from_decimal(dec, home.x_min, home.x_max, home.genes, home.f);
    }

    static void do_war(Country& c1, Country& c2, int l, double r_max = 2.0) {
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

        // Recruited warriors are temporarily removed from their countries.
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

        // Each duel keeps its winner and creates a replacement in the losing
        // country. Only surviving warriors can later become prisoners.
        for (int i = 0; i < actual_l; ++i) {
            if (*war1[i] < *war2[i]) {
                wins1++;
                survivors1.push_back(war1[i]);
                c2.population.push_back(recruit_from_duel(*war1[i], *war2[i], c2, r_max));
            } else if (*war2[i] < *war1[i]) {
                wins2++;
                survivors2.push_back(war2[i]);
                c1.population.push_back(recruit_from_duel(*war2[i], *war1[i], c1, r_max));
            } else {
                survivors1.push_back(war1[i]);
                survivors2.push_back(war2[i]);
            }
        }

        // Prisoners move toward the winning country's capital. Real-coded
        // winners can assimilate immediately; Gray conversion is performed by
        // update_individual_type() after all prisoners have been transferred.
        auto assimilate = [](Country& winner, const std::shared_ptr<Individual>& prisoner) {
            if (winner.empty()) {
                winner.population.push_back(prisoner);
                return;
            }
            const auto capital = winner.population[0]->real_x();
            auto px = prisoner->real_x();
            const double r = rand_uniform(0.35, 0.85);
            for (int d = 0; d < (int)px.size(); ++d) {
                px[d] = std::clamp(px[d] + r * (capital[d] - px[d]), winner.x_min[d], winner.x_max[d]);
            }
            if (winner.itype == IndividualType::Real) {
                auto ni = std::make_shared<RealIndividual>(std::move(px), winner.x_min, winner.x_max, winner.f);
                ni->n_ep = prisoner->n_ep;
                winner.population.push_back(std::move(ni));
            } else {
                winner.population.push_back(prisoner);
            }
        };

        // The country with more duel victories receives all surviving enemy
        // warriors. A tied war returns survivors to their original countries.
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
        std::vector<double> x_min, x_max;
        std::vector<int> genes;
        double p_min = 0.1;
        double p_max = 0.5;
        int M = 10;
        int N = 20;
        int n_min = 1;
        int n_max = 5;
        int m_min = 1;
        int m_max = 3;
        int k = 3;
        int l = 3;
        double ep_elite = 0.2;
        double ep_dead = 0.3;
        int max_mutation = 3;
        int tmax = 1000;
        double gray_percent = 0.5;

        // Exponential rank-selection pressure for crossover parents.
        // 0 means uniform selection; values around 3 match L-SRTDE's donor bias.
        double parent_rank_pressure = 3.0;

        // Crossover operator shares. Values are linearly interpolated from
        // start to end according to FEs / MaxFEs and normalized when selected.
        double real_blx_share_start   = 0.70;
        double real_blx_share_end     = 0.30;
        double real_eigen_share_start = 0.00;
        double real_eigen_share_end   = 0.00;
        double real_de_share_start    = 0.30;
        double real_de_share_end      = 0.70;

        // Fixed DE controls used before success-history adaptation is enabled.
        double de_f = 0.55;
        double de_cr = 0.90;
        double de_pbest_frac = 0.20;
        double de_pool_frac = 0.60;
        double de_pool_frac_end = 0.40;
        double de_exploitation_start_frac = 0.18;
        double de_exploit_share_start = 0.45;
        double de_exploit_share_end = 0.85;
        bool de_adaptive = true;
        int de_memory_size = 5;
        double de_f_sigma = 0.10;
        double de_cr_sigma = 0.05;

        // Linear reduction follows the L-SHADE/L-SRTDE idea, adapted to the
        // multi-country structure. Two countries are kept by default so both
        // encodings and country interactions can survive late in the run.
        bool population_reduction = false;
        int min_country_size = 4;
        int min_countries = 2;

        double gray_uniform_share_start   = 0.50;
        double gray_uniform_share_end     = 0.50;
        double gray_two_point_share_start = 0.50;
        double gray_two_point_share_end   = 0.50;
        double gray_eigen_share_start     = 0.00;
        double gray_eigen_share_end       = 0.00;

        // Fraction of the best population used to estimate the Eigen basis.
        double eigen_ps = 0.50;

        // Late Eigen-guided local search around the current global best.
        // Sigma is expressed in normalized [0,1] coordinates and decreases
        // geometrically after eigen_local_search_start_frac of the FE budget.
        bool   eigen_local_search            = false;
        double eigen_local_search_start_frac = 0.50;
        double eigen_local_sigma_start       = 0.02;
        double eigen_local_sigma_end         = 1e-8;
        double eigen_min_axis_scale          = 1e-3;
        int    eigen_local_trials            = 2;
        bool   eigen_local_stats             = false;

        // Legacy Eigen parameters from the previous binomial implementation.
        // They are intentionally kept commented out: gray_eigen_prob is replaced
        // by gray_eigen_share_start/end, gray_eigen_cr is not used by arithmetic
        // Eigen crossover, and gray_eigen_ps is replaced by the common eigen_ps
        // used by both Gray and Real individuals.
        // double gray_eigen_prob = 0.10;
        // double gray_eigen_cr   = 0.80;
        // double gray_eigen_ps   = 0.50;

        bool printing = true;

        // Probabilities of the five country actions. The order used throughout
        // the implementation is Motion, Trade, War, Epidemic, Migration.
        // Values are normalized during initialization if they do not sum to 1.
        double p_motion    = 0.25;
        double p_trade     = 0.20;
        double p_war       = 0.20;
        double p_epidemic  = 0.20;
        double p_migration = 0.15;

        bool   adaptive_actions   = true;
        bool   adaptive_reproduction_operators = true;
        double operator_alpha     = 0.10;
        double operator_pmin      = 0.05;
        double action_alpha       = 0.076; // EMA learning rate for action rewards.
        double action_pmin        = 0.05;  // Probability floor for every action.
        double action_warmup_frac = 0.14;  // Fraction of tmax used for warm-up.

        // Global diversification controls.
        int    stagnation_limit     = 25;   // Iterations without global improvement.
        double restart_country_frac = 0.15; // Worst-country fraction to rebuild.
        double migration_frac       = 0.30; // Individuals replaced during migration.

    };

    struct ActionAdaptation {
        // Reward/probability indices match the action codes used by Country.
        // 0: motion, 1: trade, 2: war, 3: epidemic, 4: migration.
        std::array<double, 5> reward = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<double, 5> probs  = {0.25, 0.20, 0.20, 0.20, 0.15};

        void update(int action_idx, double f_before, double f_after, long calls_spent, double alpha) {
            if (action_idx < 0 || action_idx >= 5) return;
            if (calls_spent <= 0) calls_spent = 1;
            // Use scale-invariant improvement of the relevant country best,
            // not average-fitness changes. This prevents an operation such as
            // 1e10 -> 1e9 on a bad individual from overwhelming a meaningful
            // elite improvement such as 2700 -> 2600.
            double improvement = std::max(0.0, f_before - f_after);
            double scale = std::abs(f_before) + std::abs(f_after) + 1.0;
            double credit = (improvement / scale) / (double)calls_spent;
            reward[action_idx] = (1.0 - alpha) * reward[action_idx] + alpha * credit;
        }

        void renormalize(double p_min) {
            double sum = 0.0;
            for (double r : reward) sum += r;
            if (sum <= 1e-15) {
                // No evidence yet: keep the configured/current probabilities
                // instead of silently reverting to unrelated hard-coded values.
                return;
            }
            // Probability matching with a guaranteed floor for every action.
            double residual = 1.0 - 5.0 * p_min;
            for (int i = 0; i < 5; ++i) {
                probs[i] = p_min + residual * (reward[i] / sum);
            }
        }
    };

    explicit CountriesAlgorithm(FuncT func, Params params)
        : p_(std::move(params)),
          calls_count_(std::make_shared<long>(0)) {
        init_internal(std::move(func));
    }

    CountriesAlgorithm(FuncT func, const Vec& x_min, const Vec& x_max, Params params)
        : p_(std::move(params)),
          calls_count_(std::make_shared<long>(0)) {
        p_.x_min = x_min;
        p_.x_max = x_max;
        init_internal(std::move(func));
    }

    void init_internal(FuncT func) {
        // Crossover shares are probabilities. Normalize start/end separately so
        // each group sums to 1; then linear interpolation preserves sum == 1
        // for every FEs / MaxFEs value.
        auto normalize_real_shares = [](double& blx, double& eigen, double& de,
                                        double fallback_blx, double fallback_eigen, double fallback_de) {
            blx = std::max(0.0, blx);
            eigen = std::max(0.0, eigen);
            de = std::max(0.0, de);
            const double sum = blx + eigen + de;
            if (sum <= 1e-15) {
                blx = fallback_blx;
                eigen = fallback_eigen;
                de = fallback_de;
                return;
            }
            blx /= sum;
            eigen /= sum;
            de /= sum;
        };

        auto normalize_gray_shares = [](double& uniform, double& two_point, double& eigen) {
            uniform = std::max(0.0, uniform);
            two_point = std::max(0.0, two_point);
            eigen = std::max(0.0, eigen);
            const double sum = uniform + two_point + eigen;
            if (sum <= 1e-15) {
                uniform = 0.05;
                two_point = 0.05;
                eigen = 0.90;
                return;
            }
            uniform /= sum;
            two_point /= sum;
            eigen /= sum;
        };

        normalize_real_shares(
            p_.real_blx_share_start, p_.real_eigen_share_start, p_.real_de_share_start,
            0.35, 0.35, 0.30);
        normalize_real_shares(
            p_.real_blx_share_end, p_.real_eigen_share_end, p_.real_de_share_end,
            0.15, 0.15, 0.70);
        normalize_gray_shares(p_.gray_uniform_share_start, p_.gray_two_point_share_start, p_.gray_eigen_share_start);
        normalize_gray_shares(p_.gray_uniform_share_end,   p_.gray_two_point_share_end,   p_.gray_eigen_share_end);

        double sum_p = p_.p_motion + p_.p_trade + p_.p_war + p_.p_epidemic + p_.p_migration;
        // Accept approximately normalized input and repair small/user-supplied
        // deviations so weighted selection always receives a valid total.
        if (sum_p <= 0.0) {
            throw std::invalid_argument("The sum of action probabilities must be positive");
        }
        if (std::abs(sum_p - 1.0) > 1e-5) {
            p_.p_motion /= sum_p;
            p_.p_trade /= sum_p;
            p_.p_war /= sum_p;
            p_.p_epidemic /= sum_p;
            p_.p_migration /= sum_p;
        }

        action_adapt_real_.probs = {p_.p_motion, p_.p_trade, p_.p_war, p_.p_epidemic, p_.p_migration};
        action_adapt_gray_.probs = action_adapt_real_.probs;
        real_operator_adapt_.probs = {p_.real_blx_share_start, p_.real_de_share_start};
        gray_operator_adapt_.probs = {p_.gray_uniform_share_start, p_.gray_two_point_share_start};

        // The shared counter survives copies/moves of the objective wrapper and
        // of CountriesAlgorithm instances stored inside std::function.
        auto counter = calls_count_;
        f_ = [counter, user_func = std::move(func)](const std::vector<double>& x) -> double {
            (*counter)++;
            return user_func(x);
        };

        if (p_.genes.empty()) {
            // A 32-bit Gray grid is the default for every problem dimension.
            p_.genes.assign(p_.x_min.size(), 32);
        }

        init_countries();
    }

    void init_countries() {
        countries_.clear();
        int gray_countries = (int)std::round(p_.gray_percent * p_.M);
        int real_countries = p_.M - gray_countries;
        countries_.reserve(p_.M);

        for (int i = 0; i < gray_countries; ++i) {
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.x_min, p_.x_max, f_, IndividualType::Gray, p_.genes
            ));
        }
        for (int i = 0; i < real_countries; ++i) {
            countries_.push_back(std::make_unique<Country>(
                p_.N, p_.x_min, p_.x_max, f_, IndividualType::Real, p_.genes
            ));
        }
    }

    std::tuple<Vec, double, long> start(
        const Vec& canonical_x, double epsilon,
        std::optional<double> y_epsilon = std::nullopt,
        std::optional<long> max_calls = std::nullopt) {

        // These arguments remain in the interface for benchmark compatibility.
        // Their stopping rules are documented below but intentionally disabled.
        (void)epsilon;
        (void)y_epsilon;
        Vec best_x;
        double best_f = std::numeric_limits<double>::infinity();
        long iteration = 0;

        reset_eigen_local_stats();

        if (!countries_.empty() && !countries_[0]->population.empty()) {
            best_x = countries_[0]->population[0]->real_x();
            best_f = countries_[0]->population[0]->f_value;
        }

        // During warm-up, actions collect reward statistics while their initial
        // probabilities remain fixed, reducing adaptation to early noise.
        long warmup_iterations = static_cast<long>(std::round(
            p_.action_warmup_frac * static_cast<double>(p_.tmax)));
        int iterations_without_improvement = 0;

        for (iteration = 1; ; ++iteration) {
            // With an explicit FE budget, MaxFEs is the primary stopping rule.
            // Population reduction lowers evaluations per generation, so a
            // fixed tmax would otherwise terminate the run far below MaxFEs.
            if (max_calls.has_value()) {
                if (*calls_count_ >= max_calls.value()) {
                    if (p_.printing) std::cout << "Max calls reached: " << *calls_count_ << std::endl;
                    print_eigen_local_stats();
                    return {best_x, best_f, iteration};
                }
            } else if (iteration > p_.tmax) {
                break;
            }

            double progress;
            if (max_calls.has_value() && max_calls.value() > 0) {
                progress = std::clamp(
                    static_cast<double>(*calls_count_) / static_cast<double>(max_calls.value()),
                    0.0, 1.0
                );
            } else {
                progress = static_cast<double>(iteration - 1) / std::max(1, p_.tmax - 1);
            }

            const int schedule_iteration = std::clamp(
                1 + static_cast<int>(std::round(progress * std::max(0, p_.tmax - 1))),
                1, std::max(1, p_.tmax)
            );

            // Motion radius decreases slowly from 2.0 to 1.2; exponent 0.6
            // deliberately preserves exploration during early iterations.
            double r_max = 2.0 - 0.8 * std::pow(progress, 0.6);

            if (countries_.size() == 1 &&
                (!p_.population_reduction || p_.min_countries > 1)) {
                split_single_country();
            }

            std::vector<Country*> ptrs;
            ptrs.reserve(countries_.size());
            for (auto& c : countries_) ptrs.push_back(c.get());

            for (auto* c : ptrs) {
                if (c->action == -1) {
                    const auto& ap = (c->itype == IndividualType::Real)
                        ? action_adapt_real_.probs : action_adapt_gray_.probs;
                    c->select_action(ptrs, ap[0], ap[1], ap[2], ap[3], ap[4]);
                }
            }

            double q_max_term = (1.0 - progress) * p_.max_mutation;

            for (size_t i = 0; i < countries_.size(); ++i) {
                auto& c = countries_[i];
                int action_index = c->action;
                if (action_index == -1) continue;

                long calls_before = *calls_count_;
                double f_before = c->best_f();
                Country* partner = nullptr;
                if (action_index == 1) partner = c->ally;
                if (action_index == 2) partner = c->enemy;
                if (partner != nullptr) f_before = std::min(f_before, partner->best_f());

                if (action_index == 0) {
                    c->do_motion(r_max);
                } else if (action_index == 1 && c->ally != nullptr) {
                    Country::do_trade(*c, *c->ally, p_.k);
                } else if (action_index == 2 && c->enemy != nullptr) {
                    Country::do_war(*c, *c->enemy, p_.l, r_max);
                } else if (action_index == 3) {
                    double p_max = (c->itype == IndividualType::Gray) ? q_max_term : p_.p_max;
                    c->do_epidemic(p_.ep_elite, p_.ep_dead, p_max, q_max_term);
                } else if (action_index == 4) {
                    c->do_migration(p_.migration_frac);
                }

                if (p_.adaptive_actions) {
                    long calls_after = *calls_count_;
                    double f_after = c->best_f();
                    if (partner != nullptr && partner->size() > 0) {
                        f_after = std::min(f_after, partner->best_f());
                    }
                    auto& adapt = (c->itype == IndividualType::Real)
                        ? action_adapt_real_ : action_adapt_gray_;
                    adapt.update(action_index, f_before, f_after,
                                 calls_after - calls_before, p_.action_alpha);
                }
            }

            // Probability matching: a zero-credit operation decays; successful
            // competitors therefore receive the freed probability mass.
            if (p_.adaptive_actions &&
                ((max_calls.has_value() && progress >= p_.action_warmup_frac) ||
                 (!max_calls.has_value() && iteration >= warmup_iterations))) {
                action_adapt_real_.renormalize(p_.action_pmin);
                action_adapt_gray_.renormalize(p_.action_pmin);
            }

            remove_empty();
            if (countries_.empty()) break;

            std::sort(countries_.begin(), countries_.end(),
                      [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });

            double f_min = countries_.front()->avg_f();
            double f_max = countries_.back()->avg_f();

            if (f_min == f_max && countries_.size() > 1) {
                restart_stagnant_countries(0.5);
                std::sort(countries_.begin(), countries_.end(),
                          [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });
                f_min = countries_.front()->avg_f();
                f_max = countries_.back()->avg_f();
            }

            // Crossover shares follow the actual evaluation budget. If start()
            // is used without MaxFEs, iteration progress is kept as a fallback.
            double crossover_progress = progress;
            if (max_calls.has_value() && max_calls.value() > 0) {
                crossover_progress = std::clamp(
                    static_cast<double>(*calls_count_) / static_cast<double>(max_calls.value()),
                    0.0,
                    1.0
                );
            }

            auto interpolate_share = [crossover_progress](double start, double end) {
                return start + crossover_progress * (end - start);
            };

            double effective_de_pool_frac = p_.de_pool_frac;
            double de_probability_floor = p_.operator_pmin;
            if (crossover_progress >= p_.de_exploitation_start_frac) {
                const double den = std::max(1e-15, 1.0 - p_.de_exploitation_start_frac);
                const double exploit_progress = std::clamp(
                    (crossover_progress - p_.de_exploitation_start_frac) / den,
                    0.0, 1.0
                );
                de_probability_floor = p_.de_exploit_share_start + exploit_progress *
                    (p_.de_exploit_share_end - p_.de_exploit_share_start);
                effective_de_pool_frac = p_.de_pool_frac + exploit_progress *
                    (p_.de_pool_frac_end - p_.de_pool_frac);
            }

            if (p_.adaptive_reproduction_operators) {
                real_operator_adapt_.renormalize(p_.operator_pmin, de_probability_floor);
                gray_operator_adapt_.renormalize(p_.operator_pmin, p_.operator_pmin);
            }
            const double real_blx_share = p_.adaptive_reproduction_operators
                ? real_operator_adapt_.probs[0] : (1.0 - de_probability_floor);
            const double real_eigen_share = 0.0;
            const double real_de_share = p_.adaptive_reproduction_operators
                ? real_operator_adapt_.probs[1] : de_probability_floor;
            const double gray_uniform_share = p_.adaptive_reproduction_operators
                ? gray_operator_adapt_.probs[0] : 0.50;
            const double gray_two_point_share = p_.adaptive_reproduction_operators
                ? gray_operator_adapt_.probs[1] : 0.50;
            const double gray_eigen_share = 0.0;

            Eigen::MatrixXd eigen_basis;
            Eigen::VectorXd eigen_values;
            const Eigen::MatrixXd* eigen_basis_ptr = nullptr;

            // Early in the run the covariance is estimated from the combined
            // elite population. Late in the run it is estimated from the best
            // country so the basis describes the local valley rather than the
            // distance between different country clusters.
            const bool use_best_country_covariance =
                crossover_progress >= p_.eigen_local_search_start_frac;

            // The same covariance basis is used by Gray and real individuals.
            if (false && p_.x_min.size() > 1 &&
                (real_eigen_share > 0.0 ||
                 gray_eigen_share > 0.0 ||
                 p_.eigen_local_search)) {
                eigen_basis = build_gray_eigen_basis(
                    p_.eigen_ps,
                    &eigen_values,
                    use_best_country_covariance
                );
                eigen_basis_ptr = &eigen_basis;
            }

            // Crossover can only recombine the spread already present between
            // parents. This late local search can make arbitrarily small steps
            // around the global best along covariance eigen-directions.
            if (false && p_.eigen_local_search &&
                eigen_basis_ptr != nullptr &&
                crossover_progress >= p_.eigen_local_search_start_frac &&
                (!max_calls.has_value() || *calls_count_ < max_calls.value())) {

                const double local_den = std::max(
                    1e-15,
                    1.0 - p_.eigen_local_search_start_frac
                );
                const double local_progress = std::clamp(
                    (crossover_progress - p_.eigen_local_search_start_frac) / local_den,
                    0.0,
                    1.0
                );

                const double sigma_start = std::max(1e-15, p_.eigen_local_sigma_start);
                const double sigma_end = std::max(1e-15, p_.eigen_local_sigma_end);
                const double global_sigma =
                    sigma_start * std::pow(sigma_end / sigma_start, local_progress);

                eigen_guided_local_search(
                    *eigen_basis_ptr,
                    eigen_values,
                    global_sigma,
                    p_.eigen_min_axis_scale,
                    p_.eigen_local_trials,
                    max_calls
                );
            }

            std::vector<std::shared_ptr<Individual>> e_individuals;
            for (auto& c : countries_) {
                if (c->size() <= 1) {
                    if (c->size() == 1) e_individuals.push_back(c->population[0]);
                    continue;
                }
                c->reproduction(
                    p_.n_min,
                    p_.n_max,
                    p_.p_min,
                    p_.p_max,
                    f_min,
                    f_max,
                    schedule_iteration,
                    p_.tmax,
                    eigen_basis_ptr,
                    real_blx_share,
                    real_eigen_share,
                    real_de_share,
                    p_.de_f,
                    p_.de_cr,
                    p_.de_pbest_frac,
                    effective_de_pool_frac,
                    p_.de_adaptive,
                    p_.de_memory_size,
                    p_.de_f_sigma,
                    p_.de_cr_sigma,
                    gray_uniform_share,
                    gray_two_point_share,
                    gray_eigen_share,
                    p_.parent_rank_pressure,
                    p_.adaptive_reproduction_operators ? &real_operator_adapt_ : nullptr,
                    p_.adaptive_reproduction_operators ? &gray_operator_adapt_ : nullptr,
                    p_.operator_alpha
                );
                c->extinction(
                    p_.m_min, p_.m_max, f_min, f_max,
                    p_.population_reduction ? std::max(1, p_.min_country_size) : 0
                );
            }

            remove_empty();

            if (!countries_.empty()) {
                for (const auto& ind : e_individuals) {
                    add_individual_to_random_country(ind);
                }
                int country_size_limit = 2 * p_.N;
                if (p_.population_reduction) {
                    const int min_country_size = std::max(4, p_.min_country_size);
                    country_size_limit = std::max(
                        min_country_size,
                        static_cast<int>(std::round(
                            static_cast<double>(p_.N) +
                            static_cast<double>(min_country_size - p_.N) * crossover_progress
                        ))
                    );
                }

                for (auto& c : countries_) {
                    c->truncate(country_size_limit);
                }

                if (p_.population_reduction) {
                    reduce_country_count(crossover_progress, country_size_limit);
                }
            }

            std::sort(countries_.begin(), countries_.end(),
                      [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

            if (countries_.empty()) break;

            if (countries_[0]->population[0]->f_value < best_f) {
                best_f = countries_[0]->population[0]->f_value;
                best_x = countries_[0]->population[0]->real_x();
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement++;
            }

            if (iterations_without_improvement >= p_.stagnation_limit && countries_.size() > 1) {
                restart_stagnant_countries(p_.restart_country_frac);
                iterations_without_improvement = 0;
            }

            if (p_.printing && iteration % 50 == 0) {
                std::cout << "Iter: " << iteration << ", Best F: " << best_f
                          << ", Calls: " << *calls_count_ << std::endl;
            }

            double dist = 0.0;
            for (size_t i = 0; i < best_x.size(); ++i) {
                double diff = best_x[i] - canonical_x[i];
                dist += diff * diff;
            }
            // Coordinate-based stopping is intentionally disabled for CEC2017:
            // the shifted optimum coordinates are hidden from the optimizer.
            // if (std::sqrt(dist) <= epsilon) {
            //     return {best_x, best_f, iteration};
            // }

            // Value-based stopping is also disabled here because canonical_x
            // is a benchmark placeholder, not necessarily the true optimum.
            // The benchmark harness evaluates success against the known bias.
            // const double canonical_y = f_(canonical_x);
            // if (y_epsilon.has_value() && std::abs(best_f - canonical_y) <= y_epsilon.value()) {
            //     return {best_x, best_f, iteration};
            // }
            // if (best_f <= canonical_y) {
            //     return {best_x, best_f, iteration};
            // }
        }

        print_eigen_local_stats();
        return {best_x, best_f, iteration};
    }

private:
    FuncT f_;
    Params p_;
    std::shared_ptr<long> calls_count_;
    std::vector<std::unique_ptr<Country>> countries_;
    ActionAdaptation action_adapt_real_;
    ActionAdaptation action_adapt_gray_;
    AdaptiveOperatorPair real_operator_adapt_;
    AdaptiveOperatorPair gray_operator_adapt_;
    size_t eigen_local_axis_cursor_ = 0;
    long eigen_local_calls_ = 0;
    long eigen_local_evaluations_ = 0;
    long eigen_local_axes_tested_ = 0;
    long eigen_local_accepted_ = 0;
    long eigen_local_plus_accepted_ = 0;
    long eigen_local_minus_accepted_ = 0;
    double eigen_local_best_before_ = std::numeric_limits<double>::infinity();
    double eigen_local_best_after_ = std::numeric_limits<double>::infinity();
    double eigen_local_total_improvement_ = 0.0;
    double eigen_local_max_improvement_ = 0.0;

    void reset_eigen_local_stats() {
        eigen_local_axis_cursor_ = 0;
        eigen_local_calls_ = 0;
        eigen_local_evaluations_ = 0;
        eigen_local_axes_tested_ = 0;
        eigen_local_accepted_ = 0;
        eigen_local_plus_accepted_ = 0;
        eigen_local_minus_accepted_ = 0;
        eigen_local_best_before_ = std::numeric_limits<double>::infinity();
        eigen_local_best_after_ = std::numeric_limits<double>::infinity();
        eigen_local_total_improvement_ = 0.0;
        eigen_local_max_improvement_ = 0.0;
    }

    void print_eigen_local_stats() const {
        if (!p_.eigen_local_stats) return;

        const double acceptance_rate =
            (eigen_local_axes_tested_ > 0)
                ? 100.0 * static_cast<double>(eigen_local_accepted_) /
                    static_cast<double>(eigen_local_axes_tested_)
                : 0.0;

        std::cout << "  Eigen local stats: {"
                  << "calls=" << eigen_local_calls_
                  << ", evaluations=" << eigen_local_evaluations_
                  << ", axes_tested=" << eigen_local_axes_tested_
                  << ", accepted=" << eigen_local_accepted_
                  << ", acceptance_percent=" << acceptance_rate
                  << ", plus_accepted=" << eigen_local_plus_accepted_
                  << ", minus_accepted=" << eigen_local_minus_accepted_
                  << ", best_before=" << eigen_local_best_before_
                  << ", best_after=" << eigen_local_best_after_
                  << ", total_improvement=" << eigen_local_total_improvement_
                  << ", max_improvement=" << eigen_local_max_improvement_
                  << "}" << std::endl;
    }

    // Build one basis from the combined population rather than from a single
    // country. With N=20, a per-country covariance matrix would be strongly
    // rank-deficient for D=30, 50, or 100. The same basis is used by Gray and
    // real Eigen crossover.
    Eigen::MatrixXd build_gray_eigen_basis(
        double ps,
        Eigen::VectorXd* eigen_values = nullptr,
        bool best_country_only = false) const {
        const int dim = static_cast<int>(p_.x_min.size());
        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim, dim);

        auto set_identity_values = [&]() {
            if (eigen_values != nullptr) {
                *eigen_values = Eigen::VectorXd::Ones(dim);
            }
        };

        if (dim <= 1) {
            set_identity_values();
            return identity;
        }

        std::vector<const Individual*> pool;

        // During late refinement use one coherent local cloud. If the best
        // country does not contain enough samples for a D-dimensional
        // covariance estimate, fall back to the combined population.
        if (best_country_only && !countries_.empty()) {
            const auto best_it = std::min_element(
                countries_.begin(),
                countries_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs->best_f() < rhs->best_f();
                }
            );

            if (best_it != countries_.end() &&
                (*best_it)->population.size() >= static_cast<size_t>(dim + 1)) {
                pool.reserve((*best_it)->population.size());
                for (const auto& individual : (*best_it)->population) {
                    if (std::isfinite(individual->f_value)) {
                        pool.push_back(individual.get());
                    }
                }
            }
        }

        if (pool.size() < static_cast<size_t>(dim + 1)) {
            pool.clear();
            size_t total_size = 0;
            for (const auto& country : countries_) {
                total_size += country->population.size();
            }
            pool.reserve(total_size);

            for (const auto& country : countries_) {
                for (const auto& individual : country->population) {
                    if (std::isfinite(individual->f_value)) {
                        pool.push_back(individual.get());
                    }
                }
            }
        }

        if (pool.size() < 2) {
            set_identity_values();
            return identity;
        }

        std::sort(
            pool.begin(),
            pool.end(),
            [](const Individual* lhs, const Individual* rhs) {
                return lhs->f_value < rhs->f_value;
            }
        );

        ps = std::clamp(ps, 0.0, 1.0);
        int elite_count = static_cast<int>(
            std::ceil(ps * static_cast<double>(pool.size()))
        );

        // Covariance needs enough samples to estimate a D-dimensional
        // orientation. Use at least D+1 points whenever the pool permits it.
        const int minimum_for_covariance = std::min(
            static_cast<int>(pool.size()),
            dim + 1
        );

        elite_count = std::max(elite_count, minimum_for_covariance);
        elite_count = std::clamp(
            elite_count,
            2,
            static_cast<int>(pool.size())
        );

        Eigen::MatrixXd samples(elite_count, dim);

        for (int i = 0; i < elite_count; ++i) {
            const auto x = pool[i]->real_x();

            for (int d = 0; d < dim; ++d) {
                const double range = p_.x_max[d] - p_.x_min[d];
                if (range > 0.0) {
                    samples(i, d) = (x[d] - p_.x_min[d]) / range;
                } else {
                    samples(i, d) = 0.0;
                }
            }
        }

        const Eigen::RowVectorXd mean = samples.colwise().mean();
        const Eigen::MatrixXd centered = samples.rowwise() - mean;
        Eigen::MatrixXd covariance =
            (centered.transpose() * centered) /
            static_cast<double>(elite_count - 1);

        if (!covariance.allFinite()) {
            set_identity_values();
            return identity;
        }

        // Stabilize the eigendecomposition for nearly collapsed populations.
        const double mean_variance = covariance.diagonal().cwiseAbs().mean();
        if (!std::isfinite(mean_variance)) {
            set_identity_values();
            return identity;
        }

        covariance.diagonal().array() +=
            1e-12 * std::max(1.0, mean_variance);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            covariance,
            Eigen::ComputeEigenvectors
        );

        if (solver.info() != Eigen::Success ||
            !solver.eigenvectors().allFinite() ||
            !solver.eigenvalues().allFinite()) {
            set_identity_values();
            return identity;
        }

        if (eigen_values != nullptr) {
            *eigen_values = solver.eigenvalues().cwiseMax(0.0);
        }

        return solver.eigenvectors();
    }

    void eigen_guided_local_search(
        const Eigen::MatrixXd& eigen_basis,
        const Eigen::VectorXd& eigen_values,
        double global_sigma,
        double min_axis_scale,
        int trials,
        std::optional<long> max_calls) {

        if (countries_.empty() || trials <= 0 || global_sigma <= 0.0) {
            return;
        }

        const int dim = static_cast<int>(p_.x_min.size());
        if (dim <= 0 ||
            eigen_basis.rows() != dim ||
            eigen_basis.cols() != dim ||
            eigen_values.size() != dim ||
            !eigen_basis.allFinite() ||
            !eigen_values.allFinite()) {
            return;
        }

        auto best_country_it = std::min_element(
            countries_.begin(),
            countries_.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs->best_f() < rhs->best_f();
            }
        );
        if (best_country_it == countries_.end() || (*best_country_it)->empty()) {
            return;
        }

        Country& best_country = **best_country_it;
        Eigen::VectorXd best_normalized(dim);

        eigen_local_calls_++;
        if (!std::isfinite(eigen_local_best_before_)) {
            eigen_local_best_before_ = best_country.population[0]->f_value;
            eigen_local_best_after_ = eigen_local_best_before_;
        }

        auto update_best_normalized = [&]() {
            const auto current_best_x = best_country.population[0]->real_x();
            for (int d = 0; d < dim; ++d) {
                const double range = p_.x_max[d] - p_.x_min[d];
                best_normalized[d] =
                    (range > 0.0) ?
                    (current_best_x[d] - p_.x_min[d]) / range : 0.0;
            }
        };

        update_best_normalized();

        // Eigenvalues control only the relative step lengths. The absolute
        // radius is controlled by global_sigma. A floor prevents a collapsed
        // covariance axis from becoming permanently frozen.
        const double max_eigen_value = eigen_values.maxCoeff();
        min_axis_scale = std::clamp(min_axis_scale, 1e-12, 1.0);

        Eigen::VectorXd axis_scale = Eigen::VectorXd::Ones(dim);
        if (std::isfinite(max_eigen_value) && max_eigen_value > 1e-30) {
            for (int j = 0; j < dim; ++j) {
                const double relative_variance =
                    std::max(0.0, eigen_values[j]) / max_eigen_value;
                axis_scale[j] = std::max(
                    std::sqrt(relative_variance),
                    min_axis_scale
                );
            }
        }

        // A full-dimensional random Gaussian step is unlikely to improve a
        // narrow ill-conditioned valley: one bad component along a sensitive
        // axis can spoil the whole trial. Instead perform a small pattern
        // search in the covariance eigenbasis. Each call examines a few axes
        // in round-robin order and evaluates both + and - directions.
        for (int trial = 0; trial < trials; ++trial) {
            if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
                break;
            }

            const int axis = static_cast<int>(eigen_local_axis_cursor_ % dim);
            eigen_local_axis_cursor_++;
            eigen_local_axes_tested_++;

            const double axis_step = global_sigma * axis_scale[axis];
            if (!(axis_step > 0.0) || !std::isfinite(axis_step)) {
                continue;
            }

            std::shared_ptr<Individual> best_candidate;
            int best_candidate_sign = 0;

            for (int sign : {-1, 1}) {
                if (max_calls.has_value() && *calls_count_ >= max_calls.value()) {
                    break;
                }

                Eigen::VectorXd candidate_normalized =
                    best_normalized +
                    static_cast<double>(sign) * axis_step * eigen_basis.col(axis);

                std::vector<double> candidate_x(dim);
                for (int d = 0; d < dim; ++d) {
                    const double range = p_.x_max[d] - p_.x_min[d];
                    const double u = std::clamp(candidate_normalized[d], 0.0, 1.0);
                    candidate_x[d] = p_.x_min[d] + u * range;
                }

                std::shared_ptr<Individual> candidate;
                if (best_country.itype == IndividualType::Gray) {
                    candidate = GrayIndividual::from_real(
                        candidate_x,
                        best_country.x_min,
                        best_country.x_max,
                        best_country.genes,
                        best_country.f
                    );
                } else {
                    candidate = std::make_shared<RealIndividual>(
                        std::move(candidate_x),
                        best_country.x_min,
                        best_country.x_max,
                        best_country.f
                    );
                }

                eigen_local_evaluations_++;

                if ((!best_candidate || candidate->f_value < best_candidate->f_value) &&
                    candidate->f_value < best_country.population[0]->f_value) {
                    best_candidate_sign = sign;
                    best_candidate = std::move(candidate);
                }
            }

            if (best_candidate) {
                const double previous_best = best_country.population[0]->f_value;
                const double improvement = previous_best - best_candidate->f_value;

                eigen_local_accepted_++;
                if (best_candidate_sign > 0) {
                    eigen_local_plus_accepted_++;
                } else if (best_candidate_sign < 0) {
                    eigen_local_minus_accepted_++;
                }
                eigen_local_total_improvement_ += improvement;
                eigen_local_max_improvement_ = std::max(
                    eigen_local_max_improvement_, improvement
                );

                best_country.population.push_back(std::move(best_candidate));
                best_country.sort_population();
                best_country.truncate(2 * p_.N);
                eigen_local_best_after_ = best_country.population[0]->f_value;

                // Continue subsequent axis probes from the newly accepted best.
                // Re-reading real_x() also accounts for Gray quantization.
                update_best_normalized();
            }
        }
    }

    void reduce_country_count(double progress, int country_size_limit) {
        if (countries_.empty()) return;

        const int initial_count = std::max(1, p_.M);
        const int min_count = std::clamp(p_.min_countries, 1, initial_count);
        const int target_count = std::clamp(
            static_cast<int>(std::round(
                static_cast<double>(initial_count) +
                static_cast<double>(min_count - initial_count) * std::clamp(progress, 0.0, 1.0)
            )),
            min_count,
            initial_count
        );

        if (static_cast<int>(countries_.size()) <= target_count) return;

        // Keep the best countries, but when at least two survive preserve one
        // country of each representation if both are currently available.
        std::sort(countries_.begin(), countries_.end(),
                  [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

        std::vector<size_t> keep;
        keep.reserve(target_count);
        for (int i = 0; i < target_count; ++i) keep.push_back(static_cast<size_t>(i));

        if (target_count >= 2) {
            auto ensure_type = [&](IndividualType type) {
                bool exists = false;
                for (const auto& c : countries_) {
                    if (c->itype == type) { exists = true; break; }
                }
                if (!exists) return;

                for (size_t idx : keep) {
                    if (countries_[idx]->itype == type) return;
                }

                size_t best_idx = countries_.size();
                for (size_t i = target_count; i < countries_.size(); ++i) {
                    if (countries_[i]->itype == type) {
                        best_idx = i;
                        break;
                    }
                }
                if (best_idx == countries_.size()) return;

                // Replace the worst currently selected survivor of the other type.
                for (size_t k = keep.size(); k-- > 0;) {
                    if (countries_[keep[k]]->itype != type) {
                        keep[k] = best_idx;
                        break;
                    }
                }
            };

            ensure_type(IndividualType::Real);
            ensure_type(IndividualType::Gray);
        }

        std::sort(keep.begin(), keep.end());
        keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

        std::vector<std::shared_ptr<Individual>> removed_leaders;
        std::vector<std::unique_ptr<Country>> survivors;
        survivors.reserve(target_count);

        for (size_t i = 0; i < countries_.size(); ++i) {
            if (std::binary_search(keep.begin(), keep.end(), i)) {
                survivors.push_back(std::move(countries_[i]));
            } else if (!countries_[i]->population.empty()) {
                // Preserve one representative of a removed basin, but discard
                // the rest so total population really decreases.
                removed_leaders.push_back(countries_[i]->population[0]->clone());
            }
        }

        countries_ = std::move(survivors);
        while (static_cast<int>(countries_.size()) > target_count) {
            countries_.pop_back();
        }

        for (const auto& leader : removed_leaders) {
            add_individual_to_random_country(leader);
        }

        for (auto& c : countries_) {
            c->truncate(country_size_limit);
        }

        for (auto& c : countries_) {
            c->action = -1;
            c->ally = nullptr;
            c->enemy = nullptr;
        }
    }

    void remove_empty() {
        // Country-level operations may eliminate every resident.
        countries_.erase(
            std::remove_if(countries_.begin(), countries_.end(),
                           [](const auto& c) { return c->empty(); }),
            countries_.end()
        );
    }

    void add_individual_to_random_country(const std::shared_ptr<Individual>& ind) {
        if (countries_.empty()) return;
        auto& rc = countries_[rand_int(0, (int)countries_.size() - 1)];
        int dim = (int)rc->x_min.size();
        std::shared_ptr<Individual> converted;

        if (ind->itype == rc->itype) {
            converted = ind->clone();
        } else if (rc->itype == IndividualType::Real) {
            // Conversion preserves both the cached objective value and the
            // individual's epidemic age, avoiding an unnecessary evaluation.
            auto rx = ind->real_x();
            auto ni = std::make_shared<RealIndividual>(std::move(rx), rc->x_min, rc->x_max, ind->f_value);
            ni->n_ep = ind->n_ep;
            converted = ni;
        } else {
            auto rx = ind->real_x();
            std::vector<uint64_t> gc(dim);
            for (int d = 0; d < dim; ++d) {
                double step = (rc->x_max[d] - rc->x_min[d]) / (double)((1ULL << rc->genes[d]) - 1);
                int64_t v = (int64_t)std::round((rx[d] - rc->x_min[d]) / step);
                uint64_t max_v = (1ULL << rc->genes[d]) - 1;
                gc[d] = tc_to_gray_code((uint64_t)std::clamp(v, (int64_t)0, (int64_t)max_v));
            }
            auto ni = std::make_shared<GrayIndividual>(gc, rc->x_min, rc->x_max, rc->genes, ind->f_value);
            ni->n_ep = ind->n_ep;
            converted = ni;
        }
        rc->population.push_back(converted);
        rc->sort_population();
    }

    void restart_stagnant_countries(double restart_fraction) {
        if (countries_.size() <= 1) return;
        // countries_ is sorted best-to-worst before this helper is called; the
        // leading country is therefore always protected from restart.
        int restart_count = std::clamp(
            (int)std::ceil(restart_fraction * (double)countries_.size()),
            1, (int)countries_.size() - 1);

        size_t start_index = countries_.size() - restart_count;
        for (size_t i = start_index; i < countries_.size(); ++i) {
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

        // Pair-based actions cannot operate with one country. Redistribute its
        // residents among smaller countries and refill short groups randomly.
        auto& original = countries_[0];
        auto all_individuals = std::move(original->population);
        countries_.clear();

        int total_individuals = (int)all_individuals.size();
        int target_size = std::max(2, p_.N / 2);
        int new_country_count = std::max(
            2, (total_individuals + target_size - 1) / target_size);

        std::shuffle(all_individuals.begin(), all_individuals.end(), rng_engine);

        int gray_country_count = (int)std::round(
            p_.gray_percent * (double)new_country_count);
        gray_country_count = std::clamp(gray_country_count, 0, new_country_count);

        int individual_offset = 0;
        for (int country_index = 0; country_index < new_country_count; ++country_index) {
            IndividualType type = (country_index < gray_country_count)
                ? IndividualType::Gray : IndividualType::Real;
            auto new_c = std::make_unique<Country>(
                target_size, p_.x_min, p_.x_max, f_, type, p_.genes
            );
            new_c->population.clear();

            int take = std::min(target_size, total_individuals - individual_offset);
            for (int i = 0; i < take; ++i) {
                new_c->population.push_back(all_individuals[individual_offset + i]);
            }
            individual_offset += take;

            while (new_c->size() < target_size) {
                new_c->population.push_back(new_c->make_random_individual());
            }

            new_c->update_individual_type();
            new_c->sort_population();
            countries_.push_back(std::move(new_c));
        }

        for (int i = individual_offset; i < total_individuals; ++i) {
            add_individual_to_random_country(all_individuals[i]);
        }
    }
};

// using CountriesAlgorithmMethod = CountriesAlgorithm;