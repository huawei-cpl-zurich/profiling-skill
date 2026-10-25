/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_MATH_HPP
#define TLA_TUPLE_MATH_HPP

#include "tla/tuple/tuple.hpp"
#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/utils/math.hpp"
#include "tla/utils/functional.hpp"
#include "tla/utils/type_traits.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

// is_int_tuple: true if T is an integral, or a tuple whose elements are all is_int_tuple (recursive)
template <class T>
struct is_int_tuple : is_integral<T> {};

template <class... Ts>
struct is_int_tuple<tla::tuple<Ts...>> : Bool<(is_int_tuple<Ts>::value && ...)> {};

template <class T>
inline constexpr bool is_int_tuple_v = is_int_tuple<T>::value;

// ---------------------------------------------------------------------------
// Product / product / size / product_like: aggregation over tuple elements
// ---------------------------------------------------------------------------

struct Product {
    template <class T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const& a) const
    {
        if constexpr (is_tuple_v<T>) {
            if constexpr (tuple_size_v<T> == 0) {
                return Int<1>{};
            } else {
                return detail::transform_apply_impl(Product{}, multiplies_unary_lfold{}, a);
            }
        } else if constexpr (is_integral_v<T>) {
            return a;
        } else {
            static_assert(dependent_false<T>, "Product: invalid operand type");
        }
    }
};

// product: multiply all elements (recursive, empty tuple = 1)
template <class T>
CATLASS_HOST_DEVICE constexpr auto product(T const& t)
{
    return Product{}(t);
}

// size: total element count (alias for product)
template <class T>
CATLASS_HOST_DEVICE constexpr auto size(T const& t)
{
    return product(t);
}

// product_like: product of leaves in t, guided by structure of g
template <class T, class TG>
CATLASS_HOST_DEVICE constexpr auto product_like(T const& t, TG const& g)
{
    return transform_leaf([](auto const&, auto const& x) { return product(x); }, g, t);
}

// product_each: product of each element (one-level flatten)
template <class T>
CATLASS_HOST_DEVICE constexpr auto product_each(T const& t)
{
    return detail::transform_impl(Product{}, wrap(t));
}

// ---------------------------------------------------------------------------
// Arithmetic operators: element-wise +, -, *, / with scalar broadcasting
// ---------------------------------------------------------------------------

#define TLA_TUPLE_BINARY_OP(OP)                                                                       \
    template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>                 \
    CATLASS_HOST_DEVICE constexpr auto operator OP(T0 const& a, T1 const& b)                          \
    {                                                                                                 \
        if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {                                             \
            TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);                                                       \
            return detail::transform_impl([](auto const& x, auto const& y) { return x OP y; }, a, b); \
        } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {                                   \
            return detail::transform_impl([&](auto const& x) { return x OP b; }, a);                  \
        } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {                                   \
            return detail::transform_impl([&](auto const& x) { return a OP x; }, b);                  \
        } else {                                                                                      \
            static_assert(dependent_false<T0, T1>, "operator" #OP ": invalid operand types");         \
        }                                                                                             \
    }

TLA_TUPLE_BINARY_OP(+);
TLA_TUPLE_BINARY_OP(-);
TLA_TUPLE_BINARY_OP(*);
TLA_TUPLE_BINARY_OP(/);

#undef TLA_TUPLE_BINARY_OP

// minimum: element-wise min (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto minimum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return minimum(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return minimum(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return minimum(t0, x); }, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return min(t0, t1);
    } else {
        static_assert(dependent_false<T0, T1>, "minimum: invalid operand types");
    }
}

// maximum: element-wise max (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto maximum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return maximum(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return maximum(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return maximum(t0, x); }, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return max(t0, t1);
    } else {
        static_assert(dependent_false<T0, T1>, "maximum: invalid operand types");
    }
}

// clip_sub: element-wise clamping subtraction (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto clip_sub(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_impl([](auto const& x, auto const& y) { return clip_sub(x, y); }, t0, t1);
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return clip_sub(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return clip_sub(t0, x); }, t1);
    } else {
        static_assert(dependent_false<T0, T1>, "clip_sub: invalid operand types");
    }
}

// ceil_div: tiling-aware ceiling division (recursive)
// - both tuples: pad shorter with 1, element-wise
// - tuple / scalar: cascading (divisor consumed across elements)
// - scalar / tuple: product then divide
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto ceil_div(T0 const& a, T1 const& b)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        static_assert(tuple_size<T0>::value >= tuple_size<T1>::value, "Mismatched ranks");
        constexpr int R = tuple_size<T0>::value;
        return detail::transform_impl(
            [](auto const& x, auto const& y) { return ceil_div(x, y); }, a, append<R>(b, Int<1>{}));
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        auto result = fold(
            [](auto const& init, auto const& ai) {
                return make_tuple(append(get<0>(init), ceil_div(ai, get<1>(init))), ceil_div(get<1>(init), ai));
            },
            make_tuple(make_tuple(), b), a);
        return get<0>(result);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return ceil_div(a, product(b));
    } else {
        static_assert(dependent_false<T0, T1>, "ceil_div: invalid operand types");
    }
}

// round_up: element-wise round up to multiple (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES((is_tuple_v<T0>) || (is_tuple_v<T1>))>
CATLASS_HOST_DEVICE constexpr auto round_up(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        static_assert(tuple_size<T0>::value >= tuple_size<T1>::value, "Mismatched ranks");
        constexpr int R = tuple_size<T0>::value;
        return detail::transform_impl(
            [](auto const& x, auto const& y) { return round_up(x, y); }, t0, append<R>(t1, Int<1>{}));
    } else if constexpr (is_tuple_v<T0> && is_integral_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return round_up(x, t1); }, t0);
    } else if constexpr (is_integral_v<T0> && is_tuple_v<T1>) {
        return detail::transform_impl([&](auto const& x) { return round_up(t0, x); }, t1);
    } else {
        static_assert(dependent_false<T0, T1>, "round_up: invalid operand types");
    }
}

// inner_product: dot product (recursive). For tuples, element-wise multiply then sum;
// for scalars, returns the product.
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto inner_product(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_apply_impl(
            [](auto const& x, auto const& y) { return inner_product(x, y); },
            [](auto const&... v) { return (Int<0>{} + ... + v); }, t0, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return t0 * t1;
    } else {
        static_assert(dependent_false<T0, T1>, "inner_product: invalid operand types");
    }
}

// evenly_divides: check if t0 is evenly divisible by t1 (element-wise)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto evenly_divides(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple_v<T0> && is_tuple_v<T1>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::transform_apply_impl(
            [](auto const& x, auto const& y) { return (x % y) == 0; }, [](auto... bs) { return (bs && ...); }, t0, t1);
    } else if constexpr (is_integral_v<T0> && is_integral_v<T1>) {
        return (t0 % t1) == 0;
    } else {
        static_assert(dependent_false<T0, T1>, "evenly_divides: invalid operand types");
    }
}

} // end namespace tla

#endif // TLA_TUPLE_MATH_HPP
