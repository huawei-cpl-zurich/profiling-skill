/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_INTEGRAL_INTEGRAL_SEQUENCE_HPP
#define TLA_INTEGRAL_INTEGRAL_SEQUENCE_HPP

#include <cstddef>

#include "tla/integral/integral_constant.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

template <typename T, T... Ns>
struct IntegerSequence {
    using value_type = T;
    static constexpr size_t size()
    {
        return sizeof...(Ns);
    }
};

template <typename Sequence, typename T, size_t N, typename = void>
struct MakeIntegerSequenceImpl;

template <typename T, typename NS>
struct MakeIntegerSequenceImpl<NS, T, 0> {
    typedef NS type;
};

template <typename T, T... Ns, size_t N>
struct MakeIntegerSequenceImpl<IntegerSequence<T, Ns...>, T, N, TLA_REQUIRES_T(N > 0)> {
    typedef typename MakeIntegerSequenceImpl<IntegerSequence<T, N - 1, Ns...>, T, N - 1>::type type;
};

template <typename T, T N>
using make_integer_sequence = typename MakeIntegerSequenceImpl<IntegerSequence<T>, T, N>::type;

// integer_sequence
template <class T, T... Ints>
using integer_sequence = IntegerSequence<T, Ints...>;

// index_sequence
template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

// int_sequence
template <int... Ints>
using int_sequence = integer_sequence<int, Ints...>;

template <int N>
using make_int_sequence = make_integer_sequence<int, N>;

// Shortcuts
template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

// make_range<Begin, End>: integer sequence Begin, Begin+1, ..., End-1
namespace detail {

template <int Begin, class S>
struct range_impl;

template <int Begin, int... Is>
struct range_impl<Begin, seq<Is...>> {
    using type = seq<(Is + Begin)...>;
};

} // end namespace detail

template <int Begin, int End>
using make_range = typename detail::range_impl<Begin, make_seq<(End > Begin) ? (End - Begin) : 0>>::type;

// make_rseq: reverse integer sequence N-1, N-2, ..., 1, 0
namespace detail {

template <int N, int... Is>
struct make_rseq_impl : make_rseq_impl<N - 1, Is..., N - 1> {};

template <int... Is>
struct make_rseq_impl<0, Is...> {
    using type = seq<Is...>;
};

} // end namespace detail

template <int N>
using make_rseq = typename detail::make_rseq_impl<N>::type;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

template <class Tuple>
using tuple_rseq = make_rseq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

// tuple traits for integer_sequence
template <class T, T... Ints>
struct tuple_size<integer_sequence<T, Ints...>> : std::integral_constant<size_t, sizeof...(Ints)> {};

template <size_t I, class T, T... Is>
struct tuple_element<I, integer_sequence<T, Is...>> {
    constexpr static T idx[sizeof...(Is)] = {Is...};
    using type = C<idx[I]>;
};

template <size_t I, class T, T... Ints>
CATLASS_HOST_DEVICE constexpr tuple_element_t<I, integer_sequence<T, Ints...>> get(integer_sequence<T, Ints...>)
{
    static_assert(I < sizeof...(Ints), "Index out of range");
    return {};
}

} // end namespace tla

#endif // TLA_INTEGRAL_INTEGRAL_SEQUENCE_HPP
