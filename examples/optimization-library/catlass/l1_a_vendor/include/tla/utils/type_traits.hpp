/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_UTILS_TYPE_TRAITS_HPP
#define TLA_UTILS_TYPE_TRAITS_HPP

#include "catlass/detail/macros.hpp"

#include <cstddef>
#include <type_traits>
#include <tuple>

// SFINAE helper macros for template constraints.
// TLA_REQUIRES: default template parameter (pointer form).
// TLA_REQUIRES_T: type form for non-type template parameters.
#define TLA_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr
#define TLA_REQUIRES_T(...) typename std::enable_if<(__VA_ARGS__)>::type

namespace tla {

// ---------------------------------------------------------------------------
// Re-exports from <type_traits>
// ---------------------------------------------------------------------------

using std::enable_if;
using std::enable_if_t;
using std::void_t;

using std::is_arithmetic;
using std::is_arithmetic_v;

using std::is_empty;
using std::is_empty_v;

using std::is_same;
using std::is_same_v;

using std::is_signed;
using std::is_signed_v;
using std::is_unsigned;
using std::is_unsigned_v;

// std::is_integral intentionally not re-exported: tla::is_integral (integral_constant.hpp) shadows it.
template <class T>
using is_std_integral = std::is_integral<T>;

// ---------------------------------------------------------------------------
// Re-exports from <utility>: declval, forward, move
// ---------------------------------------------------------------------------

template <class T>
CATLASS_HOST_DEVICE std::add_rvalue_reference_t<T> declval() noexcept;

template <class T>
CATLASS_HOST_DEVICE constexpr T&& forward(std::remove_reference_t<T>& arg) noexcept
{
    return static_cast<T&&>(arg);
}

template <class T>
CATLASS_HOST_DEVICE constexpr T&& forward(std::remove_reference_t<T>&& arg) noexcept
{
    static_assert(!std::is_lvalue_reference<T>::value, "Cannot forward an rvalue reference as an lvalue.");
    return static_cast<T&&>(arg);
}

template <class T>
CATLASS_HOST_DEVICE constexpr std::remove_reference_t<T>&& move(T&& arg) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(arg);
}

// ---------------------------------------------------------------------------
// remove_cvref (C++20 polyfill for C++17)
// ---------------------------------------------------------------------------

template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

// ---------------------------------------------------------------------------
// tuple_size / same_tuple_size: tla-local wrappers around std::tuple_size
// ---------------------------------------------------------------------------

template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<std::size_t, std::tuple_size<T>::value> {};

template <class T>
inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;

// same_tuple_size: true if all Ts have the same tuple_size as T0
template <class T0, class... Ts>
struct same_tuple_size : std::bool_constant<((tuple_size<Ts>::value == tuple_size<T0>::value) && ...)> {};

template <class... Ts>
inline constexpr bool same_tuple_size_v = same_tuple_size<Ts...>::value;

// ---------------------------------------------------------------------------
// tuple_element / tuple_element_t: tla-local wrapper around std::tuple_element
// ---------------------------------------------------------------------------

template <size_t I, class T, class = void>
struct tuple_element;

template <size_t I, class T>
struct tuple_element<I, T, std::void_t<typename std::tuple_element<I, T>::type>> : std::tuple_element<I, T> {};

template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// ---------------------------------------------------------------------------
// dependent_false: always-false value dependent on template parameters
// ---------------------------------------------------------------------------

template <class... Args>
inline constexpr bool dependent_false = false;

} // namespace tla

#define TLA_ASSERT_SAME_TUPLE_SIZE(...) static_assert(::tla::same_tuple_size_v<__VA_ARGS__>, "tuple_size mismatch")

#endif // TLA_UTILS_TYPE_TRAITS_HPP
