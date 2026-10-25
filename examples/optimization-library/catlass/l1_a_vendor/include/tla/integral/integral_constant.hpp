/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP
#define TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP

#include "catlass/detail/macros.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

// Base type: std-conforming integral constant with explicit type enforcement
template <class T, T v>
struct integral_constant {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    CATLASS_HOST_DEVICE constexpr operator value_type() const noexcept
    {
        return value;
    }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept
    {
        return value;
    }
};

// Simplified aliases: C (auto deduction), Bool (bool), Int (int), Int64 (long long)
template <auto v>
using C = integral_constant<decltype(v), v>;

// Legacy alias; prefer integral_constant or C in new code.
template <class T, T v>
using constant = C<v>;

template <bool b>
using Bool = C<b>;

template <bool b>
using bool_constant = Bool<b>;

template <int v>
using Int = C<v>;

template <long long v>
using Int64 = C<v>;

// Bool type aliases
using true_type = Bool<true>;
using false_type = Bool<false>;

// Int type aliases
#define TLA_INT_ALIAS(N) \
    using _##N = Int<N>; \
    using _##N##LL = Int64<N##LL>

TLA_INT_ALIAS(0);
TLA_INT_ALIAS(1);
TLA_INT_ALIAS(2);
TLA_INT_ALIAS(3);
TLA_INT_ALIAS(4);
TLA_INT_ALIAS(5);
TLA_INT_ALIAS(6);
TLA_INT_ALIAS(7);
TLA_INT_ALIAS(8);
TLA_INT_ALIAS(9);
TLA_INT_ALIAS(10);
TLA_INT_ALIAS(12);
TLA_INT_ALIAS(16);
TLA_INT_ALIAS(24);
TLA_INT_ALIAS(32);
TLA_INT_ALIAS(48);
TLA_INT_ALIAS(64);
TLA_INT_ALIAS(96);
TLA_INT_ALIAS(128);
TLA_INT_ALIAS(192);
TLA_INT_ALIAS(256);
TLA_INT_ALIAS(384);
TLA_INT_ALIAS(512);
TLA_INT_ALIAS(768);
TLA_INT_ALIAS(1024);
TLA_INT_ALIAS(2048);
TLA_INT_ALIAS(4096);
TLA_INT_ALIAS(8192);
TLA_INT_ALIAS(16384);
TLA_INT_ALIAS(32768);
TLA_INT_ALIAS(65536);

#undef TLA_INT_ALIAS

// Type traits: classify and test integral constant types
template <class T>
struct is_integral : Bool<is_std_integral<T>::value> {};
template <class T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};
template <class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template <class T>
struct is_static : Bool<is_empty_v<remove_cvref_t<T>>> {};
template <class T>
inline constexpr bool is_static_v = is_static<T>::value;

template <auto n, class T>
struct is_constant_base : false_type {};
template <auto n, class T, T v>
struct is_constant_base<n, integral_constant<T, v>> : Bool<v == n> {};
template <auto n, class T>
struct is_constant : is_constant_base<n, remove_cvref_t<T>> {};
template <auto n, class T>
inline constexpr bool is_constant_v = is_constant<n, T>::value;

// Underscore placeholder: empty tag for "take the whole dimension" in Coord/tensor indexing
struct Underscore {
    using type = Underscore;
};

template <>
struct is_integral<Underscore> : true_type {};

inline constexpr Underscore _{};

template <class T>
struct is_underscore : Bool<is_same_v<remove_cvref_t<T>, Underscore>> {};
template <class T>
inline constexpr bool is_underscore_v = is_underscore<T>::value;

} // namespace tla

#define TLA_STATIC_CHECK(...) static_assert(!::tla::is_constant_v<false, decltype(__VA_ARGS__)>, "static check failed.")

#endif // TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP
