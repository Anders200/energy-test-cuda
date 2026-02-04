#pragma once
#include <vector>
#include "common/point.hpp"

namespace energy_test {
namespace CUDA {

double energy_statistic(
    const std::vector<Point>& X,
    const std::vector<Point>& Y
);

} // namespace CUDA
} // namespace energy_test
