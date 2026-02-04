#pragma once
#include <vector>

namespace energy_test {

struct Point {
    std::vector<double> coords;
    std::size_t size() const { return coords.size(); }
    double operator[](std::size_t i) const { return coords[i]; }
};

} // namespace energy_test
