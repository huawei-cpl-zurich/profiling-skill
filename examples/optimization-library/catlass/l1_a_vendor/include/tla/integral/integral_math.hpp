/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_INTEGRAL_INTEGRAL_MATH_HPP
#define TLA_INTEGRAL_INTEGRAL_MATH_HPP

#include "catlass/detail/macros.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/utils/math.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

/***************/
/** Operators **/
/***************/

#define TLA_LEFT_UNARY_OP(OP)                                 \
    template <auto t>                                         \
    CATLASS_HOST_DEVICE constexpr C<(OP t)> operator OP(C<t>) \
    {                                                         \
        return {};                                            \
    }
#define TLA_BINARY_OP(OP)                                             \
    template <auto t, auto u>                                         \
    CATLASS_HOST_DEVICE constexpr C<(t OP u)> operator OP(C<t>, C<u>) \
    {                                                                 \
        return {};                                                    \
    }

TLA_LEFT_UNARY_OP(+);
TLA_LEFT_UNARY_OP(-);
TLA_LEFT_UNARY_OP(~);
TLA_LEFT_UNARY_OP(!);
TLA_LEFT_UNARY_OP(*);

TLA_BINARY_OP(+);
TLA_BINARY_OP(-);
TLA_BINARY_OP(*);
TLA_BINARY_OP(/);
TLA_BINARY_OP(%);
TLA_BINARY_OP(&);
TLA_BINARY_OP(|);
TLA_BINARY_OP(^);
TLA_BINARY_OP(<<);
TLA_BINARY_OP(>>);
// Logical overloads propagate static results; operands do not short-circuit.
TLA_BINARY_OP(&&);
TLA_BINARY_OP(||);
TLA_BINARY_OP(==);
TLA_BINARY_OP(!=);
TLA_BINARY_OP(<);
TLA_BINARY_OP(>);
TLA_BINARY_OP(<=);
TLA_BINARY_OP(>=);

#undef TLA_BINARY_OP
#undef TLA_LEFT_UNARY_OP

//
// Mixed static-dynamic special cases
// When one operand is C<t> and the other is a runtime integral,
// the result is statically known for certain values of t.
//

// 0 * x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator*(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator*(U, C<t>)
{
    return {};
}

// 0 / x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator/(C<t>, U)
{
    return {};
}

// x % (1 or -1) = 0
template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 1 || t == -1))>
CATLASS_HOST_DEVICE constexpr C<0> operator%(U, C<t>)
{
    return {};
}

// 0 % x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator%(C<t>, U)
{
    return {};
}

// 0 & x = 0
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator&(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (t == 0))>
CATLASS_HOST_DEVICE constexpr C<0> operator&(U, C<t>)
{
    return {};
}

// Like constant-constant overloads, mixed logical overloads do not short-circuit.
// false && x = false
template <auto t, class U, TLA_REQUIRES((is_std_integral<U>::value) && (!bool(t)))>
CATLASS_HOST_DEVICE constexpr C<false> operator&&(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (!bool(t)))>
CATLASS_HOST_DEVICE constexpr C<false> operator&&(U, C<t>)
{
    return {};
}

// true || x = true
template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (bool(t)))>
CATLASS_HOST_DEVICE constexpr C<true> operator||(C<t>, U)
{
    return {};
}

template <class U, auto t, TLA_REQUIRES((is_std_integral<U>::value) && (bool(t)))>
CATLASS_HOST_DEVICE constexpr C<true> operator||(U, C<t>)
{
    return {};
}

//
// Named functions from math.hpp
//

#define TLA_NAMED_BINARY_FN(OP)                                         \
    template <auto t, auto u>                                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, C<u>)                   \
    {                                                                   \
        return C<OP(t, u)>{};                                           \
    }                                                                   \
    template <auto t, class U, TLA_REQUIRES(is_std_integral<U>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, U u)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }                                                                   \
    template <class T, auto u, TLA_REQUIRES(is_std_integral<T>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(T t, C<u>)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }

TLA_NAMED_BINARY_FN(max);
TLA_NAMED_BINARY_FN(min);
TLA_NAMED_BINARY_FN(clip_sub);
TLA_NAMED_BINARY_FN(ceil_div);
TLA_NAMED_BINARY_FN(round_up);

#undef TLA_NAMED_BINARY_FN

// Variadic max/min: 2+args recursive, delegates to binary overloads.
// SFINAE on is_integral<T0> avoids conflict with tuple-aware overloads.
template <class T0, class T1, class... Ts, TLA_REQUIRES(is_integral<remove_cvref_t<T0>>::value)>
CATLASS_HOST_DEVICE constexpr auto max(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    return max(max(t0, t1), ts...);
}

template <class T0, class T1, class... Ts, TLA_REQUIRES(is_integral<remove_cvref_t<T0>>::value)>
CATLASS_HOST_DEVICE constexpr auto min(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    return min(min(t0, t1), ts...);
}

#define TLA_NAMED_UNARY_FN(OP)                  \
    template <auto t>                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>) \
    {                                           \
        return C<OP(t)>{};                      \
    }

TLA_NAMED_UNARY_FN(abs);

#undef TLA_NAMED_UNARY_FN

// conditional_return: compile-time conditional that preserves static-ness.
// When both branches are the same C<v>, returns C<v> (no runtime cost).
// When branches differ, falls back to runtime ternary.
template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr decltype(auto) conditional_return(true_type, TrueType&& t, FalseType&&)
{
    return tla::forward<TrueType>(t);
}

template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr decltype(auto) conditional_return(false_type, TrueType&&, FalseType&& f)
{
    return tla::forward<FalseType>(f);
}

template <auto v>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool, C<v> const&, C<v> const&)
{
    return C<v>{};
}

template <auto v, auto u>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool b, C<v> const&, C<u> const&)
{
    return b ? v : u;
}

template <class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr auto conditional_return(bool b, TrueType const& t, FalseType const& f)
{
    return b ? t : f;
}

template <bool b, class TrueType, class FalseType>
CATLASS_HOST_DEVICE constexpr auto conditional_return(TrueType const& t, FalseType const& f)
{
    if constexpr (b) {
        return t;
    } else {
        return f;
    }
}

} // namespace tla

#endif // TLA_INTEGRAL_INTEGRAL_MATH_HPP
