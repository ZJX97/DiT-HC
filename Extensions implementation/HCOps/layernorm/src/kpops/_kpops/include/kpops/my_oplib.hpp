#pragma once

#include <cassert>
#include <cstddef>

#if __cplusplus < 202002L
    #include <array>
#else
    #include <tuple>
    #include <utility>
#endif

namespace my_oplib
{
/**
 * @brief Compute the offset of a multi-dimensional array.
 *
 * @param args First half is the indexes, second half is the size of each
 *             dimension.
 * @return std::uint32_t The offset of the multi-dimensional array.
 *
 * @example computeOffset(1, 2, 3, 4, 5, 6) -> 3*1 + 2*6 + 1*6*5 = 45
 */
template <typename OffsetT, typename... ArgsT>
[[nodiscard]] constexpr auto computeOffset(ArgsT... args) -> OffsetT
{
    constexpr std::size_t nArgs = sizeof...(ArgsT);
    constexpr std::size_t nDims = nArgs / 2;

    OffsetT offset = 0, stride = 1;

#if __cplusplus >= 202002L
    auto params = std::make_tuple(static_cast<OffsetT>(args)...);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((I < nDims ? (offset += std::get<nDims - 1 - I>(params) * stride,
                       stride *= std::get<nArgs - 1 - I>(params))
                    : 0),
         ...);
    }(std::make_index_sequence<nDims>{});
#else
    auto params = std::array{static_cast<OffsetT>(args)...};
    for (std::size_t i = 0; i < nDims; ++i) {
        offset += params[nDims - 1 - i] * stride;
        stride *= params[nArgs - 1 - i];
    }
#endif

    return offset;
}

/**
 * @brief Calculate the ceiling of the division of two integers.
 *
 * @tparam T1 The type of the dividend.
 * @tparam T2 The type of the divisor.
 * @param a The dividend.
 * @param b The divisor.
 * @return The ceiling of the division of `a` by `b`.
 *
 * @bug I prefer to use concept for restricting T1 and T2 here, but clangd 18
 *      seems not supporting concepts for cuda yet?
 */
template <typename T1, typename T2,
          typename = std::enable_if_t<std::is_integral_v<T1> &&
                                      std::is_integral_v<T2>>>
constexpr auto ceilDiv(T1 a, T2 b) -> T1
{
    return T1((a + b - 1) / b);
}

}  // namespace my_oplib