#pragma once

#include <type_traits>

#include <tc/vec.hpp>

namespace tc {

namespace detail {

// Flat component buffer reused by the GLSL-style constructors of mat.
template <class T, int N>
struct FlatComponents {
    using value_type = T;
    T values[N]{};

    constexpr T &component(int const i) noexcept { return values[i]; }
};

} // namespace detail

// GLSL matCxR: C columns of R-component vectors, column-major like GLSL.
// Each column is a tc::vec, so the layout equals std430 (a mat3 column takes
// 16 bytes) and matrices can live in buffers and push constants unchanged.
template <class T, int C, int R>
struct mat {
    static_assert(std::is_same_v<T, float> or std::is_same_v<T, double>, "matrices hold float or double");
    static_assert(C >= 2 and C <= 4 and R >= 2 and R <= 4, "GLSL matrices have 2 to 4 columns and rows");

    using value_type = T;
    using column_type = vec<T, R>;

    constexpr mat() noexcept = default;

    // mat3(1.0f): the scalar on the diagonal, zero elsewhere.
    template <class S, std::enable_if_t<detail::ComponentSourceOf<S>::count == 1, int> = 0>
    constexpr explicit mat(S const &scalar) noexcept {
        auto flat = detail::FlatComponents<T, 1>{};
        auto cursor = 0;
        detail::ComponentSourceOf<S>::append(flat, cursor, scalar);
        for (int i = 0; i < C and i < R; ++i) {
            columns[i][i] = flat.values[0];
        }
    }

    // mat2(column0, column1) or mat2(a, b, c, d): components column by column.
    template <class... Args,
              std::enable_if_t<(sizeof...(Args) >= 2)
                                   and ((detail::ComponentSourceOf<Args>::count + ...) == C * R),
                               int> = 0>
    constexpr mat(Args const &...args) noexcept {
        auto flat = detail::FlatComponents<T, C * R>{};
        auto cursor = 0;
        (detail::ComponentSourceOf<Args>::append(flat, cursor, args), ...);
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                columns[c][r] = flat.values[c * R + r];
            }
        }
    }

    // mat3(mat4) keeps the upper-left block; missing elements come from the identity.
    template <class U, int C2, int R2, std::enable_if_t<not(C2 == C and R2 == R and std::is_same_v<U, T>), int> = 0>
    constexpr explicit mat(mat<U, C2, R2> const &other) noexcept {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                columns[c][r] = (c < C2 and r < R2) ? static_cast<T>(other[c][r]) : (c == r ? T{1} : T{0});
            }
        }
    }

    static constexpr int length() noexcept { return C; }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr column_type &operator[](I const index) noexcept {
        assert(index >= 0 and static_cast<int>(index) < C);
        return columns[index];
    }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr column_type const &operator[](I const index) const noexcept {
        assert(index >= 0 and static_cast<int>(index) < C);
        return columns[index];
    }

    friend constexpr mat operator-(mat const &a) noexcept {
        return map(a, [](column_type const &c) { return -c; });
    }
    friend constexpr mat operator+(mat const &a, mat const &b) noexcept {
        return zip(a, b, [](column_type const &l, column_type const &r) { return l + r; });
    }
    friend constexpr mat operator-(mat const &a, mat const &b) noexcept {
        return zip(a, b, [](column_type const &l, column_type const &r) { return l - r; });
    }
    friend constexpr mat operator+(mat const &a, T const s) noexcept {
        return map(a, [s](column_type const &c) { return c + s; });
    }
    friend constexpr mat operator-(mat const &a, T const s) noexcept {
        return map(a, [s](column_type const &c) { return c - s; });
    }
    friend constexpr mat operator*(mat const &a, T const s) noexcept {
        return map(a, [s](column_type const &c) { return c * s; });
    }
    friend constexpr mat operator/(mat const &a, T const s) noexcept {
        return map(a, [s](column_type const &c) { return c / s; });
    }
    friend constexpr mat operator+(T const s, mat const &a) noexcept { return a + s; }
    friend constexpr mat operator-(T const s, mat const &a) noexcept {
        return map(a, [s](column_type const &c) { return s - c; });
    }
    friend constexpr mat operator*(T const s, mat const &a) noexcept { return a * s; }
    friend constexpr mat operator/(T const s, mat const &a) noexcept {
        return map(a, [s](column_type const &c) { return s / c; });
    }

    // Linear algebra product: (C x R) * (K x C) -> (K x R).
    template <int K>
    friend constexpr mat<T, K, R> operator*(mat const &a, mat<T, K, C> const &b) noexcept {
        auto result = mat<T, K, R>{};
        for (int k = 0; k < K; ++k) {
            result[k] = a * b[k];
        }
        return result;
    }

    // Matrix times column vector.
    friend constexpr column_type operator*(mat const &m, vec<T, C> const &v) noexcept {
        auto result = column_type{};
        for (int c = 0; c < C; ++c) {
            result += m.columns[c] * v[c];
        }
        return result;
    }

    // Row vector times matrix.
    friend constexpr vec<T, C> operator*(column_type const &v, mat const &m) noexcept {
        auto result = vec<T, C>{};
        for (int c = 0; c < C; ++c) {
            auto sum = T{0};
            for (int r = 0; r < R; ++r) {
                sum += v[r] * m.columns[c][r];
            }
            result[c] = sum;
        }
        return result;
    }

    friend constexpr mat &operator+=(mat &a, mat const &b) noexcept { return a = a + b; }
    friend constexpr mat &operator-=(mat &a, mat const &b) noexcept { return a = a - b; }
    friend constexpr mat &operator*=(mat &a, T const s) noexcept { return a = a * s; }
    friend constexpr mat &operator/=(mat &a, T const s) noexcept { return a = a / s; }
    friend constexpr mat &operator*=(mat &a, mat<T, C, C> const &b) noexcept { return a = a * b; }

    friend constexpr bool operator==(mat const &a, mat const &b) noexcept {
        for (int c = 0; c < C; ++c) {
            if (a.columns[c] != b.columns[c]) {
                return false;
            }
        }
        return true;
    }
    friend constexpr bool operator!=(mat const &a, mat const &b) noexcept { return not(a == b); }

private:
    template <class F>
    static constexpr mat map(mat const &a, F const f) noexcept {
        auto result = mat{};
        for (int c = 0; c < C; ++c) {
            result.columns[c] = f(a.columns[c]);
        }
        return result;
    }

    template <class F>
    static constexpr mat zip(mat const &a, mat const &b, F const f) noexcept {
        auto result = mat{};
        for (int c = 0; c < C; ++c) {
            result.columns[c] = f(a.columns[c], b.columns[c]);
        }
        return result;
    }

    column_type columns[C]{};
};

using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;
using mat2x2 = mat<float, 2, 2>;
using mat2x3 = mat<float, 2, 3>;
using mat2x4 = mat<float, 2, 4>;
using mat3x2 = mat<float, 3, 2>;
using mat3x3 = mat<float, 3, 3>;
using mat3x4 = mat<float, 3, 4>;
using mat4x2 = mat<float, 4, 2>;
using mat4x3 = mat<float, 4, 3>;
using mat4x4 = mat<float, 4, 4>;
using dmat2 = mat<double, 2, 2>;
using dmat3 = mat<double, 3, 3>;
using dmat4 = mat<double, 4, 4>;

static_assert(sizeof(mat2) == 16 and sizeof(mat3) == 48 and sizeof(mat4) == 64);
static_assert(sizeof(mat3x2) == 24 and sizeof(mat2x3) == 32);

template <class T, int C, int R>
constexpr mat<T, R, C> transpose(mat<T, C, R> const &m) noexcept {
    auto result = mat<T, R, C>{};
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            result[r][c] = m[c][r];
        }
    }
    return result;
}

template <class T, int C, int R>
constexpr mat<T, C, R> matrixCompMult(mat<T, C, R> const &a, mat<T, C, R> const &b) noexcept {
    auto result = mat<T, C, R>{};
    for (int c = 0; c < C; ++c) {
        result[c] = a[c] * b[c];
    }
    return result;
}

// GLSL outerProduct(c, r): column vector c times row vector r.
template <class T, int R, int C>
constexpr mat<T, C, R> outerProduct(vec<T, R> const &column, vec<T, C> const &row) noexcept {
    auto result = mat<T, C, R>{};
    for (int c = 0; c < C; ++c) {
        result[c] = column * row[c];
    }
    return result;
}

template <class T>
constexpr T determinant(mat<T, 2, 2> const &m) noexcept {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template <class T>
constexpr T determinant(mat<T, 3, 3> const &m) noexcept {
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
         + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
}

namespace detail {

// The 3x3 minor of a 4x4 matrix without column `skipColumn` and row `skipRow`.
template <class T>
constexpr T minor4(mat<T, 4, 4> const &m, int const skipColumn, int const skipRow) noexcept {
    auto sub = mat<T, 3, 3>{};
    for (int c = 0, sc = 0; c < 4; ++c) {
        if (c == skipColumn) {
            continue;
        }
        for (int r = 0, sr = 0; r < 4; ++r) {
            if (r == skipRow) {
                continue;
            }
            sub[sc][sr] = m[c][r];
            ++sr;
        }
        ++sc;
    }
    return determinant(sub);
}

} // namespace detail

template <class T>
constexpr T determinant(mat<T, 4, 4> const &m) noexcept {
    auto result = T{0};
    for (int c = 0; c < 4; ++c) {
        auto const sign = c % 2 == 0 ? T{1} : T{-1};
        result += sign * m[c][0] * detail::minor4(m, c, 0);
    }
    return result;
}

template <class T>
constexpr mat<T, 2, 2> inverse(mat<T, 2, 2> const &m) noexcept {
    auto const invDet = T{1} / determinant(m);
    return mat<T, 2, 2>{m[1][1] * invDet, -m[0][1] * invDet, -m[1][0] * invDet, m[0][0] * invDet};
}

template <class T>
constexpr mat<T, 3, 3> inverse(mat<T, 3, 3> const &m) noexcept {
    auto const invDet = T{1} / determinant(m);
    auto result = mat<T, 3, 3>{};
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            // inverse[c][r] = cofactor(r, c) / det, the cofactor of row c / column r of m.
            auto const c0 = (r + 1) % 3;
            auto const c1 = (r + 2) % 3;
            auto const r0 = (c + 1) % 3;
            auto const r1 = (c + 2) % 3;
            result[c][r] = (m[c0][r0] * m[c1][r1] - m[c1][r0] * m[c0][r1]) * invDet;
        }
    }
    return result;
}

template <class T>
constexpr mat<T, 4, 4> inverse(mat<T, 4, 4> const &m) noexcept {
    auto const invDet = T{1} / determinant(m);
    auto result = mat<T, 4, 4>{};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            auto const sign = (c + r) % 2 == 0 ? T{1} : T{-1};
            result[c][r] = sign * detail::minor4(m, r, c) * invDet;
        }
    }
    return result;
}

} // namespace tc
