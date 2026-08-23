#pragma once
// ============================================================
// test_functions.hpp — полный набор из 16 стандартных 10-мерных
// бенчмарк-функций, как в abc_testing.cpp (ArtificialBeeColony).
// ============================================================
#include <cmath>
#include <vector>
#include <numeric>

namespace tf {

using Vec = std::vector<double>;

// 1. Sphere: x ∈ [-5.12, 5.12]
inline double sphere(const Vec& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

// 2. Bent Cigar: x ∈ [-100, 100]
inline double bent_cigar(const Vec& x) {
    double s = x[0] * x[0];
    for (size_t i = 1; i < x.size(); ++i) s += 1e6 * x[i] * x[i];
    return s;
}

// 3. Discus: x ∈ [-100, 100]
inline double discus(const Vec& x) {
    double s = 1e6 * x[0] * x[0];
    for (size_t i = 1; i < x.size(); ++i) s += x[i] * x[i];
    return s;
}

// 4. Zakharov: x ∈ [-5, 10]
inline double zakharov(const Vec& x) {
    double sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum1 += x[i] * x[i];
        sum2 += 0.5 * (double)(i + 1) * x[i];
    }
    return sum1 + sum2 * sum2 + sum2 * sum2 * sum2 * sum2;
}

// 5. Schwefel's Problem 1.2: x ∈ [-100, 100]
inline double schwefel_12(const Vec& x) {
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double inner = 0.0;
        for (size_t j = 0; j <= i; ++j) inner += x[j];
        s += inner * inner;
    }
    return s;
}

// 6. Schwefel's Problem 2.2: x ∈ [-100, 100]
inline double schwefel_22(const Vec& x) {
    double sum = 0.0, prod = 1.0;
    for (double v : x) { sum += std::fabs(v); prod *= std::fabs(v); }
    return sum + prod;
}

// 7. Styblinski-Tang: x ∈ [-5.12, 5.12]  (aka S-T)
inline double styblinski_tang(const Vec& x) {
    double s = 0.0;
    for (double v : x) s += (v * v * v * v - 16.0 * v * v + 5.0 * v);
    return 0.5 * s;
}

// 8. Rastrigin: x ∈ [-5.12, 5.12]
inline double rastrigin(const Vec& x) {
    double s = 10.0 * x.size();
    for (double v : x) s += v * v - 10.0 * std::cos(2 * M_PI * v);
    return s;
}

// 9. Schwefel (2.26): x ∈ [-500, 500]
inline double schwefel(const Vec& x) {
    double s = 418.9829 * x.size();
    for (double v : x) s -= v * std::sin(std::sqrt(std::fabs(v)));
    return s;
}

// 10. Happy Cat: x ∈ [-5, 5]
inline double happy_cat(const Vec& x) {
    int n = (int)x.size();
    double sum_sq = 0.0, sum = 0.0;
    for (double v : x) { sum_sq += v * v; sum += v; }
    double alpha = 0.125;
    return std::pow(std::fabs(sum_sq - n), 2 * alpha) + (0.5 * sum_sq + sum) / n + 0.5;
}

// 11. Ackley: x ∈ [-32.768, 32.768]
inline double ackley(const Vec& x) {
    int n = (int)x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (double v : x) { sum1 += v * v; sum2 += std::cos(2 * M_PI * v); }
    return -20.0 * std::exp(-0.2 * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + 20.0 + M_E;
}

// 12. Griewank: x ∈ [-600, 600]
inline double griewank(const Vec& x) {
    double sum = 0.0, prod = 1.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i] / 4000.0;
        prod *= std::cos(x[i] / std::sqrt((double)(i + 1)));
    }
    return sum - prod + 1.0;
}

// 13. Rosenbrock: x ∈ [-5, 10]
inline double rosenbrock(const Vec& x) {
    double s = 0.0;
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        s += 100.0 * t1 * t1 + t2 * t2;
    }
    return s;
}

// 14. Expanded Schaffer's F6: x ∈ [-100, 100]
inline double expanded_schaffer_f6(const Vec& x) {
    auto g = [](double a, double b) {
        double num = std::sin(std::sqrt(a * a + b * b));
        num = num * num - 0.5;
        double den = 1.0 + 0.001 * (a * a + b * b);
        den = den * den;
        return 0.5 + num / den;
    };
    double s = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double a = x[i];
        double b = (i + 1 < n) ? x[i + 1] : x[0];
        s += g(a, b);
    }
    return s;
}

// 15. Expanded Griewank plus Rosenbrock: x ∈ [-10, 10]
inline double expanded_griewank_plus_rosenbrock(const Vec& x) {
    auto rosen_pair = [](double a, double b) {
        double t1 = b - a * a;
        double t2 = 1.0 - a;
        return 100.0 * t1 * t1 + t2 * t2;
    };
    auto griewank1 = [](double z) {
        return z * z / 4000.0 - std::cos(z) + 1.0;
    };
    double s = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double a = x[i];
        double b = (i + 1 < n) ? x[i + 1] : x[0];
        s += griewank1(rosen_pair(a, b));
    }
    return s;
}

// 16. Levy: x ∈ [-10, 10]
inline double levy(const Vec& x) {
    size_t n = x.size();
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) w[i] = 1.0 + (x[i] - 1.0) / 4.0;

    double term1 = std::sin(M_PI * w[0]) * std::sin(M_PI * w[0]);
    double term3 = (w[n - 1] - 1) * (w[n - 1] - 1) *
                   (1.0 + std::sin(2 * M_PI * w[n - 1]) * std::sin(2 * M_PI * w[n - 1]));
    double sum = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        double wi = w[i];
        sum += (wi - 1) * (wi - 1) * (1.0 + 10.0 * std::sin(M_PI * wi + 1) * std::sin(M_PI * wi + 1));
    }
    return term1 + sum + term3;
}

} // namespace tf
