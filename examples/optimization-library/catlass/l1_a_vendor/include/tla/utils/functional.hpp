/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_UTILS_FUNCTIONAL_HPP
#define TLA_UTILS_FUNCTIONAL_HPP

#include "tla/utils/type_traits.hpp"
#include "tla/utils/math.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

// ---------------------------------------------------------------------------
// identity — returns input unchanged
// ---------------------------------------------------------------------------

struct identity {
    template <class T>
    CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const
    {
        return tla::forward<T>(arg);
    }
};

// ---------------------------------------------------------------------------
// Macro-generated unary functors
// ---------------------------------------------------------------------------

#define TLA_LEFT_UNARY_OP(NAME, OP)                                            \
    struct NAME {                                                              \
        template <class T>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const \
        {                                                                      \
            return (OP tla::forward<T>(arg));                                  \
        }                                                                      \
    }

#define TLA_NAMED_UNARY_OP(NAME, OP)                                           \
    struct NAME {                                                              \
        template <class T>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const \
        {                                                                      \
            return OP(tla::forward<T>(arg));                                   \
        }                                                                      \
    }

TLA_LEFT_UNARY_OP(negate, -);
TLA_NAMED_UNARY_OP(abs_fn, abs);

#undef TLA_LEFT_UNARY_OP
#undef TLA_NAMED_UNARY_OP

// ---------------------------------------------------------------------------
// Macro-generated binary functors
// ---------------------------------------------------------------------------

#define TLA_BINARY_OP(NAME, OP)                                                         \
    struct NAME {                                                                       \
        template <class T, class U>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs, U&& rhs) const \
        {                                                                               \
            return (tla::forward<T>(lhs) OP tla::forward<U>(rhs));                      \
        }                                                                               \
    }

#define TLA_NAMED_BINARY_OP(NAME, OP)                                                   \
    struct NAME {                                                                       \
        template <class T, class U>                                                     \
        CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs, U&& rhs) const \
        {                                                                               \
            return OP(tla::forward<T>(lhs), tla::forward<U>(rhs));                      \
        }                                                                               \
    }

TLA_BINARY_OP(plus, +);
TLA_BINARY_OP(minus, -);
TLA_BINARY_OP(greater, >);
TLA_BINARY_OP(greater_equal, >=);
TLA_BINARY_OP(less, <);
TLA_BINARY_OP(less_equal, <=);
TLA_NAMED_BINARY_OP(min_fn, min);
TLA_NAMED_BINARY_OP(max_fn, max);

#undef TLA_BINARY_OP
#undef TLA_NAMED_BINARY_OP

// ---------------------------------------------------------------------------
// Fold functors (left-fold only)
// ---------------------------------------------------------------------------

#define TLA_FOLD_OP(NAME, OP)                                         \
    struct NAME##_unary_lfold {                                       \
        template <class... T>                                         \
        CATLASS_HOST_DEVICE constexpr auto operator()(T&&... t) const \
        {                                                             \
            return (... OP t);                                        \
        }                                                             \
    }

TLA_FOLD_OP(multiplies, *);

#undef TLA_FOLD_OP

} // namespace tla

#endif // TLA_UTILS_FUNCTIONAL_HPP
