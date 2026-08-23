#pragma once
// ============================================================
// cec2017.hpp — C++ адаптер набора тестовых функций CEC2017
// (Single Objective Bound Constrained Optimization).
//
// Обёртка над оригинальным C-кодом Noor Awad / P.N. Suganthan
// (https://github.com/P-N-Suganthan/CEC2017-BoundContrained,
//  копия с доп. правками: https://github.com/dmolina/cec2017real).
//
// Использование:
//   - Рядом с исполняемым файлом должна лежать папка input_data/
//     (матрицы поворота, векторы сдвига, shuffle-данные).
//   - Размерность задачи (nx) берётся из x.size() при каждом вызове,
//     поэтому одна и та же функция работает для любой размерности,
//     поддерживаемой оригинальными данными: 2, 10, 20, 30, 50, 100.
//   - funcid: 1..30 (F2 исторически исключена из финальной версии
//     CEC2017, но её реализация всё равно присутствует в data-файлах
//     и коде, поэтому она доступна как cec2017::f2, просто не
//     учитывайте её в официальных сравнениях).
// ============================================================

#include <array>
#include <string>
#include <vector>
#include <stdexcept>

extern "C" {
    // Оригинальная сигнатура из cec17_test_func.c:
    // x       — входной вектор размерности nx
    // f       — выходной массив значений (mx штук, у нас всегда mx=1)
    // nx      — размерность задачи
    // mx      — число одновременно оцениваемых точек (используем 1)
    // func_num— номер тестовой функции (1..30)
    void cec17_test_func(double* x, double* f, int nx, int mx, int func_num);
}

namespace cec2017 {

using Vec = std::vector<double>;

// Известный оптимум (значение функции) для каждой функции CEC2017: F(x*) = funcid * 100.
inline double optimum_value(int funcid) {
    return static_cast<double>(funcid) * 100.0;
}

// Границы поиска для всех функций CEC2017 одинаковы: [-100, 100]^D.
constexpr double kLowerBound = -100.0;
constexpr double kUpperBound = 100.0;

namespace detail {
inline double eval(const Vec& x, int funcid) {
    if (x.empty()) throw std::invalid_argument("cec2017: пустой вектор x");
    Vec xx = x; // cec17_test_func принимает не-const double*
    double f = 0.0;
    cec17_test_func(xx.data(), &f, static_cast<int>(xx.size()), 1, funcid);
    return f;
}
} // namespace detail

// Функции-обёртки f1..f30 с сигнатурой double(const Vec&),
// совместимой с CountedFunction<double(*)(const Vec&)> из benchmark.hpp.
inline double f1(const Vec& x)  { return detail::eval(x, 1); }
inline double f2(const Vec& x)  { return detail::eval(x, 2); }  // исключена из офиц. набора
inline double f3(const Vec& x)  { return detail::eval(x, 3); }
inline double f4(const Vec& x)  { return detail::eval(x, 4); }
inline double f5(const Vec& x)  { return detail::eval(x, 5); }
inline double f6(const Vec& x)  { return detail::eval(x, 6); }
inline double f7(const Vec& x)  { return detail::eval(x, 7); }
inline double f8(const Vec& x)  { return detail::eval(x, 8); }
inline double f9(const Vec& x)  { return detail::eval(x, 9); }
inline double f10(const Vec& x) { return detail::eval(x, 10); }
inline double f11(const Vec& x) { return detail::eval(x, 11); }
inline double f12(const Vec& x) { return detail::eval(x, 12); }
inline double f13(const Vec& x) { return detail::eval(x, 13); }
inline double f14(const Vec& x) { return detail::eval(x, 14); }
inline double f15(const Vec& x) { return detail::eval(x, 15); }
inline double f16(const Vec& x) { return detail::eval(x, 16); }
inline double f17(const Vec& x) { return detail::eval(x, 17); }
inline double f18(const Vec& x) { return detail::eval(x, 18); }
inline double f19(const Vec& x) { return detail::eval(x, 19); }
inline double f20(const Vec& x) { return detail::eval(x, 20); }
inline double f21(const Vec& x) { return detail::eval(x, 21); }
inline double f22(const Vec& x) { return detail::eval(x, 22); }
inline double f23(const Vec& x) { return detail::eval(x, 23); }
inline double f24(const Vec& x) { return detail::eval(x, 24); }
inline double f25(const Vec& x) { return detail::eval(x, 25); }
inline double f26(const Vec& x) { return detail::eval(x, 26); }
inline double f27(const Vec& x) { return detail::eval(x, 27); }
inline double f28(const Vec& x) { return detail::eval(x, 28); }
inline double f29(const Vec& x) { return detail::eval(x, 29); }
inline double f30(const Vec& x) { return detail::eval(x, 30); }

using FuncPtr = double(*)(const Vec&);

struct FunctionEntry {
    int id;
    std::string name;
    FuncPtr func;
};

// Полный перечень 30 функций CEC2017 с человекочитаемыми именами.
inline const std::array<FunctionEntry, 30>& all_functions() {
    static const std::array<FunctionEntry, 30> table = { {
        {1,  "F1  Shifted and Rotated Bent Cigar",                 f1},
        {2,  "F2  Shifted and Rotated (Sum of Different Power)*",  f2},
        {3,  "F3  Shifted and Rotated Zakharov",                   f3},
        {4,  "F4  Shifted and Rotated Rosenbrock",                 f4},
        {5,  "F5  Shifted and Rotated Rastrigin",                  f5},
        {6,  "F6  Shifted and Rotated Expanded Scaffer F6",        f6},
        {7,  "F7  Shifted and Rotated Lunacek Bi_Rastrigin",       f7},
        {8,  "F8  Shifted and Rotated Non-Continuous Rastrigin",   f8},
        {9,  "F9  Shifted and Rotated Levy",                       f9},
        {10, "F10 Shifted and Rotated Schwefel",                   f10},
        {11, "F11 Hybrid Function 1 (N=3)",                        f11},
        {12, "F12 Hybrid Function 2 (N=3)",                        f12},
        {13, "F13 Hybrid Function 3 (N=3)",                        f13},
        {14, "F14 Hybrid Function 4 (N=4)",                        f14},
        {15, "F15 Hybrid Function 5 (N=4)",                        f15},
        {16, "F16 Hybrid Function 6 (N=4)",                        f16},
        {17, "F17 Hybrid Function 7 (N=5)",                        f17},
        {18, "F18 Hybrid Function 8 (N=5)",                        f18},
        {19, "F19 Hybrid Function 9 (N=5)",                        f19},
        {20, "F20 Hybrid Function 10 (N=6)",                       f20},
        {21, "F21 Composition Function 1 (N=3)",                   f21},
        {22, "F22 Composition Function 2 (N=3)",                   f22},
        {23, "F23 Composition Function 3 (N=4)",                   f23},
        {24, "F24 Composition Function 4 (N=4)",                   f24},
        {25, "F25 Composition Function 5 (N=5)",                   f25},
        {26, "F26 Composition Function 6 (N=5)",                   f26},
        {27, "F27 Composition Function 7 (N=6)",                   f27},
        {28, "F28 Composition Function 8 (N=6)",                   f28},
        {29, "F29 Composition Function 9 (N=3)",                   f29},
        {30, "F30 Composition Function 10 (N=3)",                  f30},
    } };
    return table;
}

} // namespace cec2017
