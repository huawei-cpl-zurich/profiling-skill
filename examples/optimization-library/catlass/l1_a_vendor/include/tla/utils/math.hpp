/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_UTILS_MATH_HPP
#define TLA_UTILS_MATH_HPP

#include "tla/utils/type_traits.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

template <class T, class U, TLA_REQUIRES((is_arithmetic<T>::value) && (is_arithmetic<U>::value))>
CATLASS_HOST_DEVICE constexpr auto max(T const& t, U const& u)
{
    return t < u ? u : t;
}

template <class T, class U, TLA_REQUIRES((is_arithmetic<T>::value) && (is_arithmetic<U>::value))>
CATLASS_HOST_DEVICE constexpr auto min(T const& t, U const& u)
{
    return t < u ? t : u;
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto clip_sub(T const& t, U const& u)
{
    return (t > u) ? (t - u) : T(0);
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto ceil_div(T const& a, U const& b)
{
    return (a + b - T(1)) / b;
}

template <class T, class U, TLA_REQUIRES((is_std_integral<T>::value) && (is_std_integral<U>::value))>
CATLASS_HOST_DEVICE constexpr auto round_up(T const& t, U const& u)
{
    return ceil_div(t, u) * u;
}

template <class T, TLA_REQUIRES(is_arithmetic<T>::value)>
CATLASS_HOST_DEVICE constexpr auto abs(T const& t)
{
    if constexpr (is_signed<T>::value) {
        return t < T(0) ? -t : t;
    } else {
        return t;
    }
}

} // namespace tla

#endif // TLA_UTILS_MATH_HPP
