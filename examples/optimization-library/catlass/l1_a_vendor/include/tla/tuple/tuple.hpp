/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_TUPLE_HPP
#define TLA_TUPLE_TUPLE_HPP

#include <cstddef>
#include <tuple>

#include "catlass/detail/macros.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

namespace detail {

// EBO stands for "empty base optimization."
template <size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

// Specialization for types T that are empty;
template <size_t N, class T>
struct EBO<N, T, true> {
    CATLASS_HOST_DEVICE constexpr EBO()
    {}

    CATLASS_HOST_DEVICE constexpr EBO(T const&)
    {}
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T getv(EBO<N, T, true> const&)
{
    return {};
}

// Specialization for types T that are not empty;
template <size_t N, class T>
struct EBO<N, T, false> {
    CATLASS_HOST_DEVICE constexpr EBO() : t_{}
    {}

    CATLASS_HOST_DEVICE constexpr EBO(T const& t) : t_{t}
    {}

    T t_;
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T const& getv(EBO<N, T, false> const& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T& getv(EBO<N, T, false>& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T&& getv(EBO<N, T, false>&& x)
{
    return static_cast<T&&>(x.t_);
}

// TupleBase
template <class IdxSeq, class... T>
struct TupleBase;

template <size_t... I, class... T>
struct TupleBase<index_sequence<I...>, T...> : EBO<I, T>... {
    CATLASS_HOST_DEVICE constexpr TupleBase()
    {}

    CATLASS_HOST_DEVICE constexpr TupleBase(T const&... t) : EBO<I, T>(t)...
    {}
};

} // end namespace detail

// tla::tuple class.
template <class... T>
struct tuple : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...> {
    CATLASS_HOST_DEVICE constexpr tuple()
    {}

    CATLASS_HOST_DEVICE constexpr tuple(T const&... t)
        : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...>(t...)
    {}
};

template <>
struct tuple<> {};

// get for tla::tuple
template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...> const& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>&& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(static_cast<tuple<T...>&&>(t));
}

// Integral SFINAE get: for scalar integral types, get<0> returns the value
template <size_t I, class T, TLA_REQUIRES(is_integral<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    static_assert(I == 0, "Index out of range");
    return static_cast<T&&>(t);
}

// Multi-index get: recursively index into nested tuples
template <size_t I0, size_t I1, size_t... Is, class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}

namespace detail {

template <class T>
auto has_tuple_size(T*) -> Bool<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size((T*)0)){};

template <class T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

template <class... Ts>
struct all_tuples : Bool<(is_tuple<Ts>::value && ...)> {};

template <class... Ts>
inline constexpr bool all_tuples_v = all_tuples<Ts...>::value;

template <class... T>
struct tuple_size<tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

template <class... T>
struct tuple_size<const tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

// make_tuple
template <class... T>
CATLASS_HOST_DEVICE constexpr tuple<T...> MakeTuple(T const&... t)
{
    return {t...};
}

// make_tuple (lowercase alias of MakeTuple)
template <class... T>
CATLASS_HOST_DEVICE constexpr tuple<T...> make_tuple(T const&... t)
{
    return {t...};
}

namespace detail {

template <class... T, class... U, size_t... I>
CATLASS_HOST_DEVICE constexpr bool tuple_equal_impl(tuple<T...> const& a, tuple<U...> const& b, index_sequence<I...>)
{
    return ((get<I>(a) == get<I>(b)) && ...);
}

} // namespace detail

template <class... T, class... U>
CATLASS_HOST_DEVICE constexpr bool operator==(tuple<T...> const& a, tuple<U...> const& b)
{
    static_assert(sizeof...(T) == sizeof...(U), "tuple size mismatch in operator==");
    return detail::tuple_equal_impl(a, b, make_index_sequence<sizeof...(T)>{});
}

template <class... T, class... U>
CATLASS_HOST_DEVICE constexpr bool operator!=(tuple<T...> const& a, tuple<U...> const& b)
{
    return !(a == b);
}

} // end namespace tla

#define TLA_ASSERT_ALL_TUPLES(...) static_assert(::tla::all_tuples_v<__VA_ARGS__>, "arguments must be tuples")

// Structured bindings support
namespace std {
template <class... T>
struct tuple_size<tla::tuple<T...>> : integral_constant<size_t, sizeof...(T)> {};

template <size_t I, class... T>
struct tuple_element<I, tla::tuple<T...>> {
    using type = tuple_element_t<I, tuple<T...>>;
};
} // namespace std

#endif // TLA_TUPLE_TUPLE_HPP
