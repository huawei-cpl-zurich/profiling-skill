/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_STRIDE_HPP
#define TLA_STRIDE_HPP

#include "tla/tuple/tuple.hpp"
#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/tuple/tuple_math.hpp"
#include "tla/utils/type_traits.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

template <class Shape, class Current = _1LL>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current = {});

template <class Shape, class Current = _1LL>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current = {});

namespace detail {

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_col_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_col_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return prepend(compact_col_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_row_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_row_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return append(compact_row_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

} // end namespace detail

// compact_col_major: column-major compact strides.
// stride[0] = current, stride[i] = stride[i-1] * shape[i-1].
// Supports nested shapes (recurses into tuple elements) and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple_v<Shape> && is_tuple_v<Current>) {
        return detail::transform_impl(
            [](auto const& s, auto const& c) { return compact_col_major(s, c); }, shape, current);
    } else if constexpr (is_tuple_v<Shape> && is_integral_v<Current>) {
        return detail::compact_col_major_impl(shape, current, tuple_seq<Shape>{});
    } else if constexpr (is_integral_v<Shape> && is_integral_v<Current>) {
        return current;
    } else {
        static_assert(dependent_false<Shape>, "Invalid parameters");
    }
}

// compact_row_major: row-major compact strides.
// stride[N-1] = current, stride[i] = stride[i+1] * shape[i+1].
// Supports nested shapes and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple_v<Shape> && is_tuple_v<Current>) {
        return detail::transform_impl(
            [](auto const& s, auto const& c) { return compact_row_major(s, c); }, shape, current);
    } else if constexpr (is_tuple_v<Shape> && is_integral_v<Current>) {
        return detail::compact_row_major_impl(shape, current, tuple_rseq<Shape>{});
    } else if constexpr (is_integral_v<Shape> && is_integral_v<Current>) {
        return current;
    } else {
        static_assert(dependent_false<Shape>, "Invalid parameters");
    }
}

namespace detail {

template <class FlatShape, class FlatOrder, class Order, int... Is>
CATLASS_HOST_DEVICE constexpr auto order_stride_start(
    FlatShape const& flat_shape, FlatOrder const& flat_order, Order const& order, seq<Is...>)
{
    return (_1LL{} * ... * conditional_return(get<Is>(flat_order) < order, get<Is>(flat_shape), _1LL{}));
}

template <class Shape, class Order, class FlatShape, class FlatOrder>
CATLASS_HOST_DEVICE constexpr auto compact_order_impl(
    Shape const& shape, Order const& order, FlatShape const& flat_shape, FlatOrder const& flat_order)
{
    if constexpr (is_tuple<Order>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Order);
        return detail::transform_impl(
            [&](auto const& s, auto const& o) { return compact_order_impl(s, o, flat_shape, flat_order); }, shape,
            order);
    } else {
        auto stride_start = order_stride_start(flat_shape, flat_order, order, tuple_seq<FlatShape>{});
        return compact_col_major(shape, stride_start);
    }
}

} // end namespace detail

template <class Shape, class Order>
CATLASS_HOST_DEVICE constexpr auto compact_order(Shape const& shape, Order const& order)
{
    static_assert(is_static<remove_cvref_t<Order>>::value, "Order must be all static");
    auto flat_shape = flatten_to_tuple(product_like(shape, order));
    auto flat_order = flatten_to_tuple(order);
    constexpr int n = tuple_size<remove_cvref_t<decltype(flat_order)>>::value;
    auto unique_order =
        detail::transform_impl([](auto I, auto v) { return v * Int<n>{} + I; }, make_range<0, n>{}, flat_order);
    auto new_order = unflatten(unique_order, order);
    return detail::compact_order_impl(shape, new_order, flat_shape, unique_order);
}

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple_v<Coord> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Coord, Shape, Stride);
        return detail::transform_apply_impl(
            [](auto const& c, auto const& s, auto const& d) { return crd2idx(c, s, d); },
            [](auto const&... xs) { return (_0LL{} + ... + xs); }, coord, shape, stride);
    } else if constexpr (is_integral_v<Coord> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Stride);
        if constexpr (is_constant<0, Coord>::value) {
            return _0LL{};
        } else {
            auto zipped =
                detail::transform_impl([](auto const& s, auto const& d) { return make_tuple(s, d); }, shape, stride);
            constexpr int N = tuple_size<Shape>::value;
            const auto [remaining, result] = fold(
                [](auto const& acc, auto const& sd) {
                    auto prod = product(get<0>(sd));
                    return make_tuple(
                        get<0>(acc) / prod, get<1>(acc) + crd2idx(get<0>(acc) % prod, get<0>(sd), get<1>(sd)));
                },
                make_tuple(coord, _0LL{}), take<0, N - 1>(zipped));
            const auto [s, d] = get<N - 1>(zipped);
            return result + crd2idx(remaining, s, d);
        }
    } else if constexpr (is_integral_v<Coord> && is_integral_v<Shape> && is_integral_v<Stride>) {
        return coord * stride;
    } else {
        static_assert(dependent_false<Coord>, "Invalid parameters");
    }
}

// crd2offset: alias for crd2idx (backward compatibility)
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const& coord, Shape const& shape, Stride const& stride)
{
    return crd2idx(coord, shape, stride);
}

template <class Coord, class Shape>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape)
{
    if constexpr (is_tuple_v<Coord> && is_tuple_v<Shape>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Coord, Shape);
        auto flat_coord = flatten_to_tuple(coord);
        auto flat_shape = flatten_to_tuple(product_like(shape, coord));
        auto zipped = detail::transform_impl(
            [](auto const& c, auto const& s) { return make_tuple(c, s); }, flat_coord, flat_shape);
        return fold_reverse(
            [](auto const& acc, auto const& cs) { return get<0>(cs) + get<1>(cs) * acc; }, _0LL{}, zipped);
    } else if constexpr (is_integral_v<Coord> && (is_tuple_v<Shape> || is_integral_v<Shape>)) {
        return coord;
    } else {
        static_assert(dependent_false<Coord>, "Invalid parameters");
    }
}

// idx2crd: inverse of crd2idx. Decomposes a flat index into a coordinate within <Shape, Stride>.
// Shape and Stride are assumed congruent (same tuple structure).
// - (tuple, tuple, tuple): each mode decomposed independently via element-wise recursion
// - (int, tuple, tuple): the same index is applied to every mode
// - (int, int, int): 1D base case: (idx / stride) % shape; shape == 1 short-circuits to 0
template <class Index, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto idx2crd(Index const& idx, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple_v<Index> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Index, Shape, Stride);
        return detail::transform_impl(
            [](auto const& i, auto const& s, auto const& d) { return idx2crd(i, s, d); }, idx, shape, stride);
    } else if constexpr (is_integral_v<Index> && is_tuple_v<Shape> && is_tuple_v<Stride>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Shape, Stride);
        return detail::transform_impl([&](auto const& s, auto const& d) { return idx2crd(idx, s, d); }, shape, stride);
    } else if constexpr (is_integral_v<Index> && is_integral_v<Shape> && is_integral_v<Stride>) {
        if constexpr (is_constant<1, Shape>::value) {
            return _0LL{};
        } else {
            return (idx / stride) % shape;
        }
    } else {
        static_assert(dependent_false<Index>, "Invalid parameters");
    }
}

// idx2crd: inverse of crd2idx 2-arg. Decomposes a flat index into a coordinate within Shape
// assuming compact column-major layout.
template <class Index, class Shape>
CATLASS_HOST_DEVICE constexpr auto idx2crd(Index const& idx, Shape const& shape)
{
    if constexpr (is_tuple_v<Index> && is_tuple_v<Shape>) {
        TLA_ASSERT_SAME_TUPLE_SIZE(Index, Shape);
        return detail::transform_impl([](auto const& i, auto const& s) { return idx2crd(i, s); }, idx, shape);
    } else if constexpr (is_integral_v<Index> && is_tuple_v<Shape>) {
        return idx2crd(idx, shape, compact_col_major(shape));
    } else if constexpr (is_integral_v<Index> && is_integral_v<Shape>) {
        return idx;
    } else {
        static_assert(dependent_false<Index>, "Invalid parameters");
    }
}

} // namespace tla

#endif // TLA_STRIDE_HPP
