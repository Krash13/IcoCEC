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

/** @brief Convert a normal unsigned integer to Gray code. */
static inline uint64_t tc_to_gray_code(uint64_t n) noexcept {
    return n ^ (n >> 1);
}

/** @brief Convert Gray code back to a normal unsigned integer. */
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

/** @brief Set the random seed so benchmark runs can be repeated. */
static inline void set_random_seed(std::uint64_t seed) {
    rng_engine.seed(seed);
}

/** @brief Draw a real number from a uniform distribution. */
static inline double rand_uniform(double lo, double hi) {
    return std::uniform_real_distribution<double>{lo, hi}(rng_engine);
}

/** @brief Draw a real number from a normal distribution. */
static inline double rand_normal(double mean = 0.0, double stddev = 1.0) {
    return std::normal_distribution<double>{mean, stddev}(rng_engine);
}

/** @brief Draw an integer from an inclusive uniform range. */
static inline int rand_int(int lo, int hi_inclusive) {
    return std::uniform_int_distribution<int>{lo, hi_inclusive}(rng_engine);
}

/** @brief Draw an unsigned integer from an inclusive uniform range. */
static inline uint64_t rand_uint64(uint64_t lo, uint64_t hi_inclusive) {
    return std::uniform_int_distribution<uint64_t>{lo, hi_inclusive}(rng_engine);
}

// ============================================================================
// Weighted country-action selection
// 0: Motion, 1: Trade, 2: War, 3: Epidemic, 4: Migration
// ============================================================================

/** @brief Choose one country action according to the given probabilities. */
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

// Adaptive probability update for two competing reproduction operators.
// A failed operator receives zero credit, so its EMA reward decays and its
// probability is automatically redistributed to the competitor.
struct AdaptiveOperatorPair {
    std::array<double, 2> reward = {1e-6, 1e-6}; ///< Smoothed usefulness score for each reproduction operator.
    std::array<double, 2> probs  = {0.5, 0.5}; ///< Current probability of choosing each reproduction operator.

    /** @brief Update the recent usefulness score of one reproduction operator. */
    void update(int idx, double credit, double alpha) {
        if (idx < 0 || idx >= 2) return;
        alpha = std::clamp(alpha, 0.0, 1.0);
        credit = std::max(0.0, credit);
        reward[idx] = (1.0 - alpha) * reward[idx] + alpha * credit;
    }

    /** @brief Convert two operator rewards into valid selection probabilities. */
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

/** @brief Give credit only when a child reaches the current near-best region. */
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
    std::vector<double> x_min, x_max; ///< Lower and upper bounds of all coordinates.
    double f_value = std::numeric_limits<double>::infinity(); ///< Cached objective value of this individual.
    // Number of epidemic mutations already applied to this individual. It
    // reduces mutation amplitude and the Gray-code bit-flip count over time.
    int n_ep = 0; ///< Number of epidemic mutations already applied to this individual.
    IndividualType itype; ///< Representation used by this individual.

    /** @brief Create the common state of an individual. */
    Individual(std::vector<double> x_min, std::vector<double> x_max, IndividualType t)
        : x_min(std::move(x_min)), x_max(std::move(x_max)), itype(t) {}

    /** @brief Destroy an individual through the base interface. */
    virtual ~Individual() = default;
    /** @brief Return the represented point in real coordinates. */
    virtual std::vector<double> real_x() const = 0;
    /** @brief Create an independent copy of this individual. */
    virtual std::shared_ptr<Individual> clone() const = 0;

    /** @brief Compare individuals by objective value. */
    bool operator<(const Individual& o) const noexcept { return f_value < o.f_value; }
    /** @brief Compare individuals by objective value. */
    bool operator>(const Individual& o) const noexcept { return f_value > o.f_value; }
    /** @brief Compare individuals by objective value. */
    bool operator<=(const Individual& o) const noexcept { return f_value <= o.f_value; }
};

// ============================================================================
// Gray Individual
// ============================================================================

struct GrayIndividual : Individual {
    std::vector<int> genes; ///< Number of Gray-code bits used for each coordinate.
    std::vector<uint64_t> code; ///< Gray-code value of each coordinate.
    std::vector<double> steps; ///< Real coordinate step represented by one integer code step.

    /** @brief Create a Gray individual and evaluate its objective value. */
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

    /** @brief Create a Gray individual from a cached objective value without a new evaluation. */
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

    /** @brief Return this individual as real-valued coordinates. */
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

    /** @brief Decode all Gray-code coordinates to integer grid positions. */
    std::vector<uint64_t> decimal_x() const {
        int dim = (int)genes.size();
        std::vector<uint64_t> dec(dim);
        for (int i = 0; i < dim; ++i) {
            dec[i] = gray_code_to_tc(code[i]);
        }
        return dec;
    }

    /** @brief Create a Gray individual from integer grid coordinates. */
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

    /** @brief Create a Gray individual from real-valued coordinates. */
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

    /** @brief Create an independent copy without a new objective evaluation. */
    std::shared_ptr<Individual> clone() const override {
        auto copy = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
        copy->n_ep = n_ep;
        return copy;
    }

    /** @brief Create a Gray-code child by flipping a scheduled number of distinct bits. */
    std::shared_ptr<GrayIndividual> mutate(double q_max_term, const FuncT& func) const {
        // 1. Calculate how many bits this individual may change now.
        // The number becomes smaller with time and after survived epidemics.
        int n = std::max(0, (int)std::floor(q_max_term - n_ep));
        // 2. If no bit must change, return a copy and reuse its fitness.
        // The genotype is unchanged, so a new function evaluation is not needed.
        if (n == 0) {
            auto copy = std::make_shared<GrayIndividual>(code, x_min, x_max, genes, f_value);
            return copy;
        }

        // 3. Put all coordinate codes into one bit sequence.
        // This lets mutation choose bits from the whole Gray genotype.
        int total_bits = 0;
        for (int g : genes) total_bits += g;

        std::vector<uint8_t> bits(total_bits);
        int pos = 0;
        for (int i = 0; i < (int)genes.size(); ++i) {
            for (int b = genes[i] - 1; b >= 0; --b) {
                bits[pos++] = (uint8_t)((code[i] >> b) & 1);
            }
        }

        // 4. Choose different bit positions and flip each selected bit once.
        // The partial shuffle prevents selecting the same bit two times.
        std::vector<int> positions(total_bits);
        std::iota(positions.begin(), positions.end(), 0);
        // Partial Fisher-Yates shuffle selects different N="flips" bits to flip
        int flips = std::min(n, total_bits);
        for (int i = 0; i < flips; ++i) {
            int j = rand_int(i, total_bits - 1);
            std::swap(positions[i], positions[j]);
            bits[positions[i]] ^= 1;
        }

        // 5. Build the Gray code of every coordinate from the changed bits.
        std::vector<uint64_t> new_code(genes.size(), 0);
        pos = 0;
        for (int i = 0; i < (int)genes.size(); ++i) {
            uint64_t v = 0;
            for (int b = genes[i] - 1; b >= 0; --b) {
                v = (v << 1) | (uint64_t)bits[pos++];
            }
            new_code[i] = v;
        }

        // 6. Create the child, decode its coordinates, and calculate fitness.
        auto child = std::make_shared<GrayIndividual>(new_code, x_min, x_max, genes, func);
        return child;
    }

    /** @brief Create two Gray-code children with uniform or two-point crossover. */
    // The caller chooses which crossover type to use. This function only
    // performs the selected bit exchange and evaluates the two new children.
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

private:
    /** @brief Precompute the real coordinate step represented by one Gray integer step. */
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
    std::vector<double> x; ///< Real-valued coordinates of the individual.

    /** @brief Create a real-valued individual and evaluate its objective value. */
    RealIndividual(std::vector<double> x,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   const FuncT& func)
        : Individual(x_min, x_max, IndividualType::Real), x(std::move(x)) {
        f_value = func(this->x);
    }

    /** @brief Create a real individual from a cached objective value without a new evaluation. */
    RealIndividual(std::vector<double> x,
                   const std::vector<double>& x_min,
                   const std::vector<double>& x_max,
                   double cached_f_value)
        : Individual(x_min, x_max, IndividualType::Real), x(std::move(x)) {
        f_value = cached_f_value;
    }

    /** @brief Return this individual as real-valued coordinates. */
    std::vector<double> real_x() const override { return x; }

    /** @brief Create an independent copy without a new objective evaluation. */
    std::shared_ptr<Individual> clone() const override {
        auto copy = std::make_shared<RealIndividual>(x, x_min, x_max, f_value);
        copy->n_ep = n_ep;
        return copy;
    }

    /** @brief Recalculate the objective value after real coordinates change. */
    void update_f(const FuncT& func) {
        f_value = func(x);
    }

    /** @brief Create one DE trial vector with current-to-pbest/1 mutation and binomial crossover. */
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

                //reflection
                if (value < x_min[d])
                    value = x_min[d] + (x_min[d] - value);
                if (value > x_max[d])
                    value = x_max[d] - (value - x_max[d]);
                
                // Match the competition DE implementations: still infeasible
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

    /** @brief Mutate a real individual and recalculate its objective value. */
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

    /** @brief Create a real-valued child with BLX-alpha crossover. */
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

        // For each coordinate i, we find the min and max value for two parents.
        // lo[i] = min(ax[i], bx[i]), hi[i] = max(ax[i], bx[i]).
        const Eigen::ArrayXd lo = ax.min(bx);
        const Eigen::ArrayXd hi = ax.max(bx);
        // The interval between two parents in each dimension.
        const Eigen::ArrayXd interval = hi - lo;
        // expand this interval by a factor α.
        // lower[i] = lo[i] - α * interval[i], upper[i] = hi[i] + α * interval[i]
        // When α is 0, the child is sampled from [lo, hi]
        const Eigen::ArrayXd lower = lo - alpha * interval;
        const Eigen::ArrayXd upper = hi + alpha * interval;

        Eigen::ArrayXd u(dim);
        for (Eigen::Index i = 0; i < dim; ++i) {
            u(i) = rand_uniform(0.0, 1.0);
        }

        // Sample the child vector from the expanded interval
        // and clamp all the coordinates to the allowed search boundaries
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
    std::vector<std::shared_ptr<Individual>> population; ///< Individuals currently living in the country.
    std::vector<double> x_min, x_max; ///< Lower and upper bounds of all coordinates.
    std::vector<int> genes; ///< Number of Gray-code bits used for each coordinate.
    FuncT f; ///< Objective function used to evaluate new individuals.
    int N; ///< Initial population size of this country.
    IndividualType itype; ///< Real/binary representation used by this individual.

    int action = -1; ///< Action selected for the current iteration; -1 means not selected.
    Country* ally = nullptr; ///< Second country used by a trade action.
    Country* enemy = nullptr; ///< Second country used by a war action.

    // Success-history state for the Real DE branch. Kept per country because
    // each country can occupy a different basin and therefore prefer a
    // different mutation/crossover scale.
    std::vector<double> de_memory_f; ///< Successful DE scale-factor values remembered by this country.
    std::vector<double> de_memory_cr; ///< Successful DE crossover-rate values remembered by this country.
    int de_memory_index = 0; ///< Position that will be updated next in the DE memory.
    double de_success_rate = 0.5; ///< Recent fraction of useful DE trials.

    /** @brief Create one country and generate its initial local population. */
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

    /** @brief Return true when the country has no individuals. */
    bool empty() const noexcept { return population.empty(); }
    /** @brief Return the current number of individuals in the country. */
    int size() const noexcept { return (int)population.size(); }

    /** @brief Sort the country so the best objective value is stored first. */
    void sort_population() {
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a->f_value < b->f_value; });
    }

    /** @brief Return the best objective value in this country. */
    double best_f() const noexcept {
        return population.empty() ? std::numeric_limits<double>::infinity() : population[0]->f_value;
    }

    /** @brief Return the average objective value in this country. */
    double avg_f() const noexcept {
        if (population.empty()) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (const auto& ind : population) s += ind->f_value;
        return s / (double)population.size();
    }

    /** @brief Select a reproduction parent with preference for better ranks. */
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

    /** @brief Choose this country action and reserve a partner for Trade or War when needed. */
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

    /** @brief Move non-leading individuals toward the current country leader. */
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

    /** @brief Remove weak individuals and mutate the non-protected survivors. */
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

    /** @brief Replace the worst part of a country with new random individuals. */
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

    /** @brief Create new individuals using the reproduction operators enabled for this country. */
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
            const double alpha_blx = std::clamp(
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
                    // cannot distort the DE search direction.
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
                        // Cauchy random change so occasional larger
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
                    // huge improvement of a very bad target (for example
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
                    auto child = RealIndividual::crossover(*a, *b, alpha_blx, x_min, x_max, f);
                    if (real_operator_adapt != nullptr) {
                        real_operator_adapt->update(
                            0,
                            normalized_elite_credit(de_elite_threshold, child->f_value),
                            operator_alpha);
                    }
                    population.push_back(std::move(child));
                } else {
                    // Eigen crossover is disabled; any unexpected selection case
                    // is repaired to BLX so covariance never re-enters the run.
                    auto child = RealIndividual::crossover(*a, *b, alpha_blx, x_min, x_max, f);
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

    /** @brief Remove a scheduled number of weak individuals from this country. */
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

    /** @brief Limit the population to the requested maximum size. */
    void truncate(int max_size) {
        if (size() > max_size) {
            population.resize(max_size);
        }
    }

    /** @brief Convert imported individuals to the representation used by this country. */
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

    /** @brief Create one random individual using this country representation. */
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

    /** @brief Exchange randomly selected individuals between two countries. */
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

    /** @brief Create a replacement individual near a duel winner after War. */
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

        // Generate one random noise direction for the entire individual.
        std::vector<double> noise(dim);
        double nrm2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            noise[d] = rand_uniform(-1.0, 1.0);
            nrm2 += noise[d] * noise[d];
        }
        const double nrm = std::sqrt(nrm2) + 1e-12;

        // Reflect beyond the winner and add a small small random side shift;
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

    /** @brief Run pairwise duels and move surviving prisoners after a war. */
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
        std::vector<double> x_min, x_max; ///< Lower and upper bounds of all coordinates.
        std::vector<int> genes; ///< Number of Gray-code bits used for each coordinate.
        double p_min = 0.1; ///< Minimum BLX-alpha value used by real crossover.
        double p_max = 0.5; ///< Maximum BLX-alpha value used by real crossover and real mutation.
        int M = 10; ///< Initial number of countries.
        int N = 20; ///< Initial number of individuals in each country.
        int n_min = 1; ///< Minimum reproduction amount for one country.
        int n_max = 5; ///< Maximum reproduction amount for one country.
        int m_min = 1; ///< Minimum number removed during extinction.
        int m_max = 3; ///< Maximum number removed during extinction.
        int k = 3; ///< Number of individuals exchanged during trade.
        int l = 3; ///< Number of duel pairs used during war.
        double ep_elite = 0.2; ///< Best fraction protected during an epidemic.
        double ep_dead = 0.3; ///< Worst fraction removed during an epidemic.
        int max_mutation = 3; ///< Maximum number of Gray bits that an epidemic may flip.
        int tmax = 1000; ///< Iteration scale used when no explicit FE budget is given.
        double gray_percent = 0.5; ///< Fraction of initial countries that use Gray-code individuals.

        // Exponential rank-selection pressure for crossover parents.
        // 0 means uniform selection; values around 3 match L-SRTDE's donor bias.
        double parent_rank_pressure = 3.0; ///< Strength of rank bias when reproduction parents are selected.

        // Crossover operator shares. Values are linearly interpolated from
        // start to end according to FEs / MaxFEs and normalized when selected.
        double real_blx_share_start   = 0.70; ///< Initial probability share of BLX crossover.
        double real_blx_share_end     = 0.30; ///< Final probability share of BLX crossover.
        double real_eigen_share_start = 0.00; ///< Initial Eigen crossover share; kept at zero in the clean version.
        double real_eigen_share_end   = 0.00; ///< Final Eigen crossover share; kept at zero in the clean version.
        double real_de_share_start    = 0.30; ///< Initial DE reproduction share.
        double real_de_share_end      = 0.70; ///< Final DE reproduction share.

        // Fixed DE controls used before success-history adaptation is enabled.
        double de_f = 0.55; ///< Default DE scale factor F.
        double de_cr = 0.90; ///< Default DE binomial crossover rate CR.
        double de_pbest_frac = 0.20; ///< Fraction of best DE candidates that may be chosen as pbest.
        double de_pool_frac = 0.60; ///< Initial fraction of the real population available to DE.
        double de_pool_frac_end = 0.40; ///< Final fraction of the real population available to DE.
        double de_exploitation_start_frac = 0.18; ///< FE fraction after which DE receives a growing probability floor.
        double de_exploit_share_start = 0.45; ///< DE probability floor when the exploitation phase begins.
        double de_exploit_share_end = 0.85; ///< DE probability floor near the end of the FE budget.
        bool de_adaptive = true; ///< Enable success-history adaptation of F and CR.
        int de_memory_size = 5; ///< Number of successful F/CR records kept in DE memory.
        double de_f_sigma = 0.10; ///< Spread used when sampling a new F around its memory value.
        double de_cr_sigma = 0.05; ///< Spread used when sampling a new CR around its memory value.

        // Linear reduction follows the L-SHADE/L-SRTDE idea, adapted to the
        // multi-country structure. Two countries are kept by default so both
        // encodings and country interactions can survive late in the run.
        bool population_reduction = true; ///< Enable gradual reduction of country and population sizes.
        int min_country_size = 4; ///< Smallest country size allowed by population reduction.
        int min_countries = 2; ///< Smallest number of countries allowed by population reduction.

        double gray_uniform_share_start   = 0.50; ///< Initial share of uniform Gray crossover.
        double gray_uniform_share_end     = 0.50; ///< Final share of uniform Gray crossover.
        double gray_two_point_share_start = 0.50; ///< Initial share of two-point Gray crossover.
        double gray_two_point_share_end   = 0.50; ///< Final share of two-point Gray crossover.
        double gray_eigen_share_start     = 0.00; ///< Initial Gray Eigen share; kept at zero in the clean version.
        double gray_eigen_share_end       = 0.00; ///< Final Gray Eigen share; kept at zero in the clean version.

        // Fraction of the best population used to estimate the Eigen basis.
        double eigen_ps = 0.50; ///< Best-population fraction formerly used to build an Eigen basis.

        // Late Eigen-guided local search around the current global best.
        // Sigma is expressed in normalized [0,1] coordinates and decreases
        // geometrically after eigen_local_search_start_frac of the FE budget.
        bool   eigen_local_search            = false; ///< Enable the old Eigen local search; disabled in the clean version.
        double eigen_local_search_start_frac = 0.50; ///< FE fraction at which old Eigen local search would start.
        double eigen_local_sigma_start       = 0.02; ///< Initial normalized step of old Eigen local search.
        double eigen_local_sigma_end         = 1e-8; ///< Final normalized step of old Eigen local search.
        double eigen_min_axis_scale          = 1e-3; ///< Minimum relative Eigen-axis scale in the old local search.
        int    eigen_local_trials            = 2; ///< Number of trials made by the old Eigen local search.
        bool   eigen_local_stats             = false; ///< Print statistics for the old Eigen local search.

        // Legacy Eigen parameters from the previous binomial implementation.
        // They are intentionally kept commented out: gray_eigen_prob is replaced
        // by gray_eigen_share_start/end, gray_eigen_cr is not used by arithmetic
        // Eigen crossover, and gray_eigen_ps is replaced by the common eigen_ps
        // used by both Gray and Real individuals.
        // double gray_eigen_prob = 0.10;
        // double gray_eigen_cr   = 0.80;
        // double gray_eigen_ps   = 0.50;

        /// Known best objective value for benchmarks such as CEC2017.
        /// Leave empty for ordinary problems where this value is not known.
        std::optional<double> target_f = std::nullopt;

        /// Print progress information to the console.
        bool printing = true;

        // Probabilities of the five country actions. The order used throughout
        // the implementation is Motion, Trade, War, Epidemic, Migration.
        // Values are normalized during initialization if they do not sum to 1.
        double p_motion    = 0.25; ///< Base probability of Motion.
        double p_trade     = 0.20; ///< Base probability of Trade.
        double p_war       = 0.20; ///< Base probability of War.
        double p_epidemic  = 0.20; ///< Base probability of Epidemic.
        double p_migration = 0.15; ///< Base probability of Migration.

        bool   adaptive_actions   = true; ///< Adapt probabilities of the five country actions.
        bool   adaptive_reproduction_operators = true; ///< Adapt probabilities of reproduction operators.
        double operator_alpha     = 0.10; ///< Learning rate for reproduction-operator rewards.
        double operator_pmin      = 0.05; ///< Minimum probability kept for each reproduction operator.
        double action_alpha       = 0.076; // EMA learning rate for action rewards. ///< Learning rate for country-action rewards.
        double action_pmin        = 0.05;  // Probability floor for every action. ///< Minimum probability kept for every country action.
        double action_warmup_frac = 0.14;  // Fraction of tmax used for warm-up. ///< Early FE fraction used before action probabilities start adapting.

        // Global diversification controls.
        int    stagnation_limit     = 25;   // Iterations without global improvement. ///< Iterations without a new global best before a restart is allowed.
        double restart_country_frac = 0.15; // Worst-country fraction to rebuild. ///< Fraction of worst countries rebuilt after stagnation.
        double migration_frac       = 0.30; // Individuals replaced during migration. ///< Fraction of a country replaced during migration.

    };

    struct ActionAdaptation {
        // Reward/probability indices match the action codes used by Country.
        // 0: motion, 1: trade, 2: war, 3: epidemic, 4: migration.
        std::array<double, 5> reward = {0.0, 0.0, 0.0, 0.0, 0.0}; ///< Smoothed usefulness score of Motion, Trade, War, Epidemic, and Migration.
        std::array<double, 5> probs  = {0.25, 0.20, 0.20, 0.20, 0.15}; ///< Current probabilities of the five country actions.

        /** @brief Update the recent reward of one country action. */
        void update(int action_idx, double f_before, double f_after, long calls_spent, double alpha) {
            if (action_idx < 0 || action_idx >= 5) return;
            if (calls_spent <= 0) calls_spent = 1;
            // Use relative improvement of the relevant country best,
            // not average-fitness changes. This prevents an operation such as
            // 1e10 -> 1e9 on a bad individual from overwhelming a meaningful
            // elite improvement such as 2700 -> 2600.
            double improvement = std::max(0.0, f_before - f_after);
            double scale = std::abs(f_before) + std::abs(f_after) + 1.0;
            double credit = (improvement / scale) / (double)calls_spent;
            reward[action_idx] = (1.0 - alpha) * reward[action_idx] + alpha * credit;
        }

        /** @brief Turn country-action rewards into probabilities with a minimum floor. */
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

    /** @brief Create the algorithm from an objective function and settings. */
    explicit CountriesAlgorithm(FuncT func, Params params)
        : params(std::move(params)),
          calls_count(std::make_shared<long>(0)) {
        init_internal(std::move(func));
    }

    /** @brief Create the algorithm and override search bounds in the supplied settings. */
    CountriesAlgorithm(FuncT func, const Vec& x_min, const Vec& x_max, Params params)
        : params(std::move(params)),
          calls_count(std::make_shared<long>(0)) {
        params.x_min = x_min;
        params.x_max = x_max;
        init_internal(std::move(func));
    }

    /** @brief Validate settings, wrap the objective counter, and create the initial countries. */
    void init_internal(FuncT user_func) {
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
            params.real_blx_share_start, params.real_eigen_share_start, params.real_de_share_start,
            0.35, 0.35, 0.30);
        normalize_real_shares(
            params.real_blx_share_end, params.real_eigen_share_end, params.real_de_share_end,
            0.15, 0.15, 0.70);
        normalize_gray_shares(params.gray_uniform_share_start, params.gray_two_point_share_start, params.gray_eigen_share_start);
        normalize_gray_shares(params.gray_uniform_share_end,   params.gray_two_point_share_end,   params.gray_eigen_share_end);

        double sum_p = params.p_motion + params.p_trade + params.p_war + params.p_epidemic + params.p_migration;
        // Accept approximately normalized input and repair small/user-supplied
        // deviations so weighted selection always receives a valid total.
        if (sum_p <= 0.0) {
            throw std::invalid_argument("The sum of action probabilities must be positive");
        }
        if (std::abs(sum_p - 1.0) > 1e-5) {
            params.p_motion /= sum_p;
            params.p_trade /= sum_p;
            params.p_war /= sum_p;
            params.p_epidemic /= sum_p;
            params.p_migration /= sum_p;
        }

        action_adapt_real.probs = {params.p_motion, params.p_trade, params.p_war, params.p_epidemic, params.p_migration};
        action_adapt_gray.probs = action_adapt_real.probs;
        real_operator_adapt.probs = {params.real_blx_share_start, params.real_de_share_start};
        gray_operator_adapt.probs = {params.gray_uniform_share_start, params.gray_two_point_share_start};

        // The shared counter survives copies/moves of the objective wrapper and
        // of CountriesAlgorithm instances stored inside std::function.
        auto counter = calls_count;
        func = [counter, objective = std::move(user_func)](const std::vector<double>& x) -> double {
            (*counter)++;
            return objective(x);
        };

        if (params.genes.empty()) {
            // A 32-bit Gray grid is the default for every problem dimension.
            params.genes.assign(params.x_min.size(), 32);
        }

        init_countries();
    }

    /** @brief Create the initial Gray and real-valued countries. */
    void init_countries() {
        countries.clear();
        int gray_countries = (int)std::round(params.gray_percent * params.M);
        int real_countries = params.M - gray_countries;
        countries.reserve(params.M);

        for (int i = 0; i < gray_countries; ++i) {
            countries.push_back(std::make_unique<Country>(
                params.N, params.x_min, params.x_max, func, IndividualType::Gray, params.genes
            ));
        }
        for (int i = 0; i < real_countries; ++i) {
            countries.push_back(std::make_unique<Country>(
                params.N, params.x_min, params.x_max, func, IndividualType::Real, params.genes
            ));
        }
    }

    /** @brief Run the Countries Algorithm until the FE limit, iteration limit, or known target value is reached. */
    std::tuple<Vec, double, long> start(
        const Vec& canonical_x, double epsilon,
        std::optional<double> y_epsilon = std::nullopt,
        std::optional<long> max_calls = std::nullopt) {

        // The coordinate tolerance is kept for compatibility with generic tests.
        // CEC2017 hides the optimum coordinates, so this algorithm does not use it.
        (void)epsilon;
        Vec best_x;
        double best_f = std::numeric_limits<double>::infinity();
        long iteration = 0;

        reset_eigen_local_stats();

        if (!countries.empty() && !countries[0]->population.empty()) {
            best_x = countries[0]->population[0]->real_x();
            best_f = countries[0]->population[0]->f_value;
        }

        // During warm-up, actions collect reward statistics while their initial
        // probabilities remain fixed, reducing adaptation to early noise.
        long warmup_iterations = static_cast<long>(std::round(
            params.action_warmup_frac * static_cast<double>(params.tmax)));
        int iterations_without_improvement = 0;

        for (iteration = 1; ; ++iteration) {
            // With an explicit FE budget, MaxFEs is the primary stopping rule.
            // Population reduction lowers evaluations per generation, so a
            // fixed tmax would otherwise terminate the run far below MaxFEs.
            // 1. Stop before a new iteration if the FE budget is already exhausted.
            if (max_calls.has_value()) {
                if (*calls_count >= max_calls.value()) {
                    if (params.printing) std::cout << "Max calls reached: " << *calls_count << std::endl;
                    print_eigen_local_stats();
                    return {best_x, best_f, iteration};
                }
            } else if (iteration > params.tmax) {
                break;
            }

            // 2. Convert the used budget to a 0..1 progress value for schedules.
            double progress;
            if (max_calls.has_value() && max_calls.value() > 0) {
                progress = std::clamp(
                    static_cast<double>(*calls_count) / static_cast<double>(max_calls.value()),
                    0.0, 1.0
                );
            } else {
                progress = static_cast<double>(iteration - 1) / std::max(1, params.tmax - 1);
            }

            const int schedule_iteration = std::clamp(
                1 + static_cast<int>(std::round(progress * std::max(0, params.tmax - 1))),
                1, std::max(1, params.tmax)
            );

            // Motion radius decreases slowly from 2.0 to 1.2; exponent 0.6
            // deliberately preserves exploration during early iterations.
            double r_max = 2.0 - 0.8 * std::pow(progress, 0.6);

            // 3. Keep more than one country when country interactions are still enabled.
            if (countries.size() == 1 &&
                (!params.population_reduction || params.min_countries > 1)) {
                split_single_country();
            }

            bool apply_ICO_actions = false;
            if (apply_ICO_actions)
            {
                // 4. Select one country action for every country that is free this round.
                std::vector<Country*> ptrs;
                ptrs.reserve(countries.size());
                for (auto& c : countries) ptrs.push_back(c.get());

                for (auto* c : ptrs) {
                    if (c->action == -1) {
                        const auto& ap = (c->itype == IndividualType::Real)
                            ? action_adapt_real.probs : action_adapt_gray.probs;
                        c->select_action(ptrs, ap[0], ap[1], ap[2], ap[3], ap[4]);
                    }
                }

                // 5. Reduce the Gray mutation strength as the evaluation budget is used.
                double q_max_term = (1.0 - progress) * params.max_mutation;

                for (size_t i = 0; i < countries.size(); ++i) {
                    auto& c = countries[i];
                    int action_index = c->action;
                    if (action_index == -1) continue;

                    long calls_before = *calls_count;
                    double f_before = c->best_f();
                    Country* partner = nullptr;
                    if (action_index == 1) partner = c->ally;
                    if (action_index == 2) partner = c->enemy;
                    if (partner != nullptr) f_before = std::min(f_before, partner->best_f());

                    if (action_index == 0) {
                        c->do_motion(r_max);
                    } else if (action_index == 1 && c->ally != nullptr) {
                        Country::do_trade(*c, *c->ally, params.k);
                    } else if (action_index == 2 && c->enemy != nullptr) {
                        Country::do_war(*c, *c->enemy, params.l, r_max);
                    } else if (action_index == 3) {
                        double p_max = (c->itype == IndividualType::Gray) ? q_max_term : params.p_max;
                        c->do_epidemic(params.ep_elite, params.ep_dead, p_max, q_max_term);
                    } else if (action_index == 4) {
                        c->do_migration(params.migration_frac);
                    }

                    if (params.adaptive_actions) {
                        long calls_after = *calls_count;
                        double f_after = c->best_f();
                        if (partner != nullptr && partner->size() > 0) {
                            f_after = std::min(f_after, partner->best_f());
                        }
                        auto& adapt = (c->itype == IndividualType::Real)
                            ? action_adapt_real : action_adapt_gray;
                        adapt.update(action_index, f_before, f_after,
                                    calls_after - calls_before, params.action_alpha);
                    }
                }

                // Probability matching: a zero-credit operation decays; successful
                // competitors therefore receive the freed probability mass.
                if (params.adaptive_actions &&
                    ((max_calls.has_value() && progress >= params.action_warmup_frac) ||
                    (!max_calls.has_value() && iteration >= warmup_iterations))) {
                    action_adapt_real.renormalize(params.action_pmin);
                    action_adapt_gray.renormalize(params.action_pmin);
                }
            }

            remove_empty();
            if (countries.empty()) break;

            std::sort(countries.begin(), countries.end(),
                      [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });

            double f_min = countries.front()->avg_f();
            double f_max = countries.back()->avg_f();

            if (f_min == f_max && countries.size() > 1) {
                restart_stagnant_countries(0.5);
                std::sort(countries.begin(), countries.end(),
                          [](const auto& a, const auto& b) { return a->avg_f() < b->avg_f(); });
                f_min = countries.front()->avg_f();
                f_max = countries.back()->avg_f();
            }

            // Crossover shares follow the actual evaluation budget. If start()
            // is used without MaxFEs, iteration progress is kept as a fallback.
            // 8. Update reproduction shares from the current FE progress.
            double crossover_progress = progress;
            if (max_calls.has_value() && max_calls.value() > 0) {
                crossover_progress = std::clamp(
                    static_cast<double>(*calls_count) / static_cast<double>(max_calls.value()),
                    0.0,
                    1.0
                );
            }

            auto interpolate_share = [crossover_progress](double start, double end) {
                return start + crossover_progress * (end - start);
            };

            double effective_de_pool_frac = params.de_pool_frac;
            double de_probability_floor = params.operator_pmin;
            if (crossover_progress >= params.de_exploitation_start_frac) {
                const double den = std::max(1e-15, 1.0 - params.de_exploitation_start_frac);
                const double exploit_progress = std::clamp(
                    (crossover_progress - params.de_exploitation_start_frac) / den,
                    0.0, 1.0
                );
                de_probability_floor = params.de_exploit_share_start + exploit_progress *
                    (params.de_exploit_share_end - params.de_exploit_share_start);
                effective_de_pool_frac = params.de_pool_frac + exploit_progress *
                    (params.de_pool_frac_end - params.de_pool_frac);
            }

            if (params.adaptive_reproduction_operators) {
                real_operator_adapt.renormalize(params.operator_pmin, de_probability_floor);
                gray_operator_adapt.renormalize(params.operator_pmin, params.operator_pmin);
            }
            const double real_blx_share = params.adaptive_reproduction_operators
                ? real_operator_adapt.probs[0] : (1.0 - de_probability_floor);
            const double real_eigen_share = 0.0;
            const double real_de_share = params.adaptive_reproduction_operators
                ? real_operator_adapt.probs[1] : de_probability_floor;
            const double gray_uniform_share = params.adaptive_reproduction_operators
                ? gray_operator_adapt.probs[0] : 0.50;
            const double gray_two_point_share = params.adaptive_reproduction_operators
                ? gray_operator_adapt.probs[1] : 0.50;
            const double gray_eigen_share = 0.0;

            Eigen::MatrixXd eigen_basis;
            Eigen::VectorXd eigen_values;
            const Eigen::MatrixXd* eigen_basis_ptr = nullptr;

            // 9. Reproduce stronger countries, remove weak residents, and keep single survivors.
            std::vector<std::shared_ptr<Individual>> e_individuals;
            for (auto& c : countries) {
                if (c->size() <= 1) {
                    if (c->size() == 1) e_individuals.push_back(c->population[0]);
                    continue;
                }
                c->reproduction(
                    params.n_min,
                    params.n_max,
                    params.p_min,
                    params.p_max,
                    f_min,
                    f_max,
                    schedule_iteration,
                    params.tmax,
                    eigen_basis_ptr,
                    real_blx_share,
                    real_eigen_share,
                    real_de_share,
                    params.de_f,
                    params.de_cr,
                    params.de_pbest_frac,
                    effective_de_pool_frac,
                    params.de_adaptive,
                    params.de_memory_size,
                    params.de_f_sigma,
                    params.de_cr_sigma,
                    gray_uniform_share,
                    gray_two_point_share,
                    gray_eigen_share,
                    params.parent_rank_pressure,
                    params.adaptive_reproduction_operators ? &real_operator_adapt : nullptr,
                    params.adaptive_reproduction_operators ? &gray_operator_adapt : nullptr,
                    params.operator_alpha
                );
                c->extinction(
                    params.m_min, params.m_max, f_min, f_max,
                    params.population_reduction ? std::max(1, params.min_country_size) : 0
                );
            }

            remove_empty();

            if (!countries.empty()) {
                for (const auto& ind : e_individuals) {
                    add_individual_to_random_country(ind);
                }
                int country_size_limit = 2 * params.N;
                if (params.population_reduction) {
                    const int min_country_size = std::max(4, params.min_country_size);
                    country_size_limit = std::max(
                        min_country_size,
                        static_cast<int>(std::round(
                            static_cast<double>(params.N) +
                            static_cast<double>(min_country_size - params.N) * crossover_progress
                        ))
                    );
                }

                for (auto& c : countries) {
                    c->truncate(country_size_limit);
                }

                if (params.population_reduction) {
                    reduce_country_count(crossover_progress, country_size_limit);
                }
            }

            std::sort(countries.begin(), countries.end(),
                      [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

            if (countries.empty()) break;

            // 10. Update the global best solution and the stagnation counter.
            if (countries[0]->population[0]->f_value < best_f) {
                best_f = countries[0]->population[0]->f_value;
                best_x = countries[0]->population[0]->real_x();
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement++;
            }

            if (iterations_without_improvement >= params.stagnation_limit && countries.size() > 1) {
                restart_stagnant_countries(params.restart_country_frac);
                iterations_without_improvement = 0;
            }

            if (params.printing && iteration % 50 == 0) {
                std::cout << "Iter: " << iteration << ", Best F: " << best_f
                          << ", Calls: " << *calls_count << std::endl;
            }

            // CEC2017 does not reveal the optimum coordinates, so coordinate
            // distance is not a valid stopping rule for these shifted functions.
            (void)canonical_x;

            // Stop early when the best objective value reaches the known target.
            // For CEC2017, set params.target_f to the function bias (100 for F1,
            // 300 for F3, and so on) and pass y_epsilon = 1e-8.
            // 11. Finish successfully when the known objective target is accurate enough.
            if (params.target_f.has_value() && y_epsilon.has_value()) {
                const double objective_error = std::abs(best_f - params.target_f.value());
                if (objective_error <= y_epsilon.value()) {
                    if (params.printing) {
                        std::cout << "Target objective reached: error=" << objective_error
                                  << ", calls=" << *calls_count << std::endl;
                    }
                    print_eigen_local_stats();
                    return {best_x, best_f, iteration};
                }
            }
        }

        print_eigen_local_stats();
        return {best_x, best_f, iteration};
    }

private:
    FuncT func; ///< Objective function wrapper that also counts function evaluations.
    Params params; ///< Algorithm settings used by this run.
    std::shared_ptr<long> calls_count; ///< Shared number of objective-function evaluations.
    std::vector<std::unique_ptr<Country>> countries; ///< All countries that are still active.
    ActionAdaptation action_adapt_real; ///< Action rewards and probabilities for real-valued countries.
    ActionAdaptation action_adapt_gray; ///< Action rewards and probabilities for Gray-code countries.
    AdaptiveOperatorPair real_operator_adapt; ///< Adaptive choice between BLX and DE for real-valued reproduction.
    AdaptiveOperatorPair gray_operator_adapt; ///< Adaptive choice between uniform and two-point Gray crossover.
    size_t eigen_local_axis_cursor = 0; ///< Next Eigen axis used by the disabled local-search statistics code.
    long eigen_local_calls = 0; ///< Number of local-search calls.
    long eigen_local_evaluations = 0; ///< Number of objective evaluations used by local search.
    long eigen_local_axes_tested = 0; ///< Number of Eigen axes tested by local search.
    long eigen_local_accepted = 0; ///< Number of accepted local-search moves.
    long eigen_local_plus_accepted = 0; ///< Accepted moves in the positive axis direction.
    long eigen_local_minus_accepted = 0; ///< Accepted moves in the negative axis direction.
    double eigen_local_best_before = std::numeric_limits<double>::infinity(); ///< Best value before local-search measurements.
    double eigen_local_best_after = std::numeric_limits<double>::infinity(); ///< Best value after local-search measurements.
    double eigen_local_total_improvement = 0.0; ///< Total improvement produced by measured local-search moves.
    double eigen_local_max_improvement = 0.0; ///< Largest improvement produced by one measured local-search move.

    /** @brief Reset counters used by the disabled Eigen local-search diagnostics. */
    void reset_eigen_local_stats() {
        eigen_local_axis_cursor = 0;
        eigen_local_calls = 0;
        eigen_local_evaluations = 0;
        eigen_local_axes_tested = 0;
        eigen_local_accepted = 0;
        eigen_local_plus_accepted = 0;
        eigen_local_minus_accepted = 0;
        eigen_local_best_before = std::numeric_limits<double>::infinity();
        eigen_local_best_after = std::numeric_limits<double>::infinity();
        eigen_local_total_improvement = 0.0;
        eigen_local_max_improvement = 0.0;
    }

    /** @brief Print Eigen local-search diagnostics when that output is enabled. */
    void print_eigen_local_stats() const {
        if (!params.eigen_local_stats) return;

        const double acceptance_rate =
            (eigen_local_axes_tested > 0)
                ? 100.0 * static_cast<double>(eigen_local_accepted) /
                    static_cast<double>(eigen_local_axes_tested)
                : 0.0;

        std::cout << "  Eigen local stats: {"
                  << "calls=" << eigen_local_calls
                  << ", evaluations=" << eigen_local_evaluations
                  << ", axes_tested=" << eigen_local_axes_tested
                  << ", accepted=" << eigen_local_accepted
                  << ", acceptance_percent=" << acceptance_rate
                  << ", plus_accepted=" << eigen_local_plus_accepted
                  << ", minus_accepted=" << eigen_local_minus_accepted
                  << ", best_before=" << eigen_local_best_before
                  << ", best_after=" << eigen_local_best_after
                  << ", total_improvement=" << eigen_local_total_improvement
                  << ", max_improvement=" << eigen_local_max_improvement
                  << "}" << std::endl;
    }

    /** @brief Reduce country count gradually when population reduction is enabled. */
    void reduce_country_count(double progress, int country_size_limit) {
        if (countries.empty()) return;

        const int initial_count = std::max(1, params.M);
        const int min_count = std::clamp(params.min_countries, 1, initial_count);
        const int target_count = std::clamp(
            static_cast<int>(std::round(
                static_cast<double>(initial_count) +
                static_cast<double>(min_count - initial_count) * std::clamp(progress, 0.0, 1.0)
            )),
            min_count,
            initial_count
        );

        if (static_cast<int>(countries.size()) <= target_count) return;

        // Keep the best countries, but when at least two survive preserve one
        // country of each representation if both are currently available.
        std::sort(countries.begin(), countries.end(),
                  [](const auto& a, const auto& b) { return a->best_f() < b->best_f(); });

        std::vector<size_t> keep;
        keep.reserve(target_count);
        for (int i = 0; i < target_count; ++i) keep.push_back(static_cast<size_t>(i));

        if (target_count >= 2) {
            auto ensure_type = [&](IndividualType type) {
                bool exists = false;
                for (const auto& c : countries) {
                    if (c->itype == type) { exists = true; break; }
                }
                if (!exists) return;

                for (size_t idx : keep) {
                    if (countries[idx]->itype == type) return;
                }

                size_t best_idx = countries.size();
                for (size_t i = target_count; i < countries.size(); ++i) {
                    if (countries[i]->itype == type) {
                        best_idx = i;
                        break;
                    }
                }
                if (best_idx == countries.size()) return;

                // Replace the worst currently selected survivor of the other type.
                for (size_t k = keep.size(); k-- > 0;) {
                    if (countries[keep[k]]->itype != type) {
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

        for (size_t i = 0; i < countries.size(); ++i) {
            if (std::binary_search(keep.begin(), keep.end(), i)) {
                survivors.push_back(std::move(countries[i]));
            } else if (!countries[i]->population.empty()) {
                // Preserve one representative of a removed basin, but discard
                // the rest so total population really decreases.
                removed_leaders.push_back(countries[i]->population[0]->clone());
            }
        }

        countries = std::move(survivors);
        while (static_cast<int>(countries.size()) > target_count) {
            countries.pop_back();
        }

        for (const auto& leader : removed_leaders) {
            add_individual_to_random_country(leader);
        }

        for (auto& c : countries) {
            c->truncate(country_size_limit);
        }

        for (auto& c : countries) {
            c->action = -1;
            c->ally = nullptr;
            c->enemy = nullptr;
        }
    }

    /** @brief Delete countries that no longer contain any individuals. */
    void remove_empty() {
        // Country-level operations may eliminate every resident.
        countries.erase(
            std::remove_if(countries.begin(), countries.end(),
                           [](const auto& c) { return c->empty(); }),
            countries.end()
        );
    }

    /** @brief Insert one individual into a random country, converting its representation if needed. */
    void add_individual_to_random_country(const std::shared_ptr<Individual>& ind) {
        if (countries.empty()) return;
        auto& rc = countries[rand_int(0, (int)countries.size() - 1)];
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

    /** @brief Rebuild a fraction of weak countries after long global stagnation. */
    void restart_stagnant_countries(double restart_fraction) {
        if (countries.size() <= 1) return;
        // countries is sorted best-to-worst before this helper is called; the
        // leading country is therefore always protected from restart.
        int restart_count = std::clamp(
            (int)std::ceil(restart_fraction * (double)countries.size()),
            1, (int)countries.size() - 1);

        size_t start_index = countries.size() - restart_count;
        for (size_t i = start_index; i < countries.size(); ++i) {
            auto& c = countries[i];
            int target_n = std::max(c->size(), params.N);
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

    /** @brief Split one remaining country so country-level interactions can continue. */
    void split_single_country() {
        if (countries.size() != 1) return;

        // Pair-based actions cannot operate with one country. Redistribute its
        // residents among smaller countries and refill short groups randomly.
        auto& original = countries[0];
        auto all_individuals = std::move(original->population);
        countries.clear();

        int total_individuals = (int)all_individuals.size();
        int target_size = std::max(2, params.N / 2);
        int new_country_count = std::max(
            2, (total_individuals + target_size - 1) / target_size);

        std::shuffle(all_individuals.begin(), all_individuals.end(), rng_engine);

        int gray_country_count = (int)std::round(
            params.gray_percent * (double)new_country_count);
        gray_country_count = std::clamp(gray_country_count, 0, new_country_count);

        int individual_offset = 0;
        for (int country_index = 0; country_index < new_country_count; ++country_index) {
            IndividualType type = (country_index < gray_country_count)
                ? IndividualType::Gray : IndividualType::Real;
            auto new_c = std::make_unique<Country>(
                target_size, params.x_min, params.x_max, func, type, params.genes
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
            countries.push_back(std::move(new_c));
        }

        for (int i = individual_offset; i < total_individuals; ++i) {
            add_individual_to_random_country(all_individuals[i]);
        }
    }
};

// using CountriesAlgorithmMethod = CountriesAlgorithm;