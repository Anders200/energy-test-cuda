#include "binder.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using energy_test::BenchmarkConfig;
    using energy_test::Method;

    BenchmarkConfig base;
    base.seed = 42;
    base.dim = 20;
    base.permutations = 199;
    base.delta = 0.2;

    std::vector<std::size_t> sample_sizes = {100, 500, 2000, 10000};
    std::vector<std::size_t> dims = {20};
    std::vector<Method> methods = {Method::CPU_BASELINE, Method::CPU_CROSS_DIST};

    auto results = energy_test::run_benchmarks(sample_sizes, dims, base, methods);

    std::size_t current_dim = 0;
    for (const auto& r : results) {
        if (r.dim != current_dim) {
            current_dim = r.dim;
            std::cout << "\n=== Dimension " << current_dim << " ===\n";
        }

        std::cout
            << std::setw(6) << r.n
            << " | dim=" << r.dim
            << " | " << (r.method == Method::CPU_BASELINE ? "CPU stat=  " : "CROSS stat=")
            << std::setw(10) << r.statistic
            << " | p=" << std::setw(8) << r.p_value
            << " | stat_t=" << std::fixed << std::setprecision(4) << r.stat_seconds << "s"
            << " | p_t=" << r.p_value_seconds << "s"
            << '\n';

        if (r.method == Method::CPU_CROSS_DIST) {
            std::cout << "\n";
        }
    }

    std::cout << "------------" << std::endl;

    return 0;
}
