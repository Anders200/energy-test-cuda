#include "binder.hpp"

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    using energy_test::BenchmarkConfig;
    using energy_test::BenchmarkResult;
    using energy_test::Method;

    // =========================
    // Base configuration
    // =========================
    BenchmarkConfig base;
    base.seed = 42;
    base.dim = 2;              // CUDA backend = dim=2
    base.permutations = 50;    // mniej, żeby było szybko
    base.delta = 0.2;
    base.warmup = true;

    // =========================
    // Test sizes
    // =========================
    std::vector<std::size_t> sample_sizes = {
        256, 512, 1024, 2048
    };

    std::vector<std::size_t> dims = {2};

    // =========================
    // Methods to test
    // =========================
    std::vector<Method> methods = {
        Method::CPU_BASELINE,
        Method::CPU_CROSS_DIST,
        Method::CUDA_CROSS_DIST
    };

    // =========================
    // Run benchmarks
    // =========================
    auto results = energy_test::run_benchmarks(
        sample_sizes,
        dims,
        base,
        methods
    );

    // =========================
    // Print results
    // =========================
    std::size_t current_dim = 0;
    double last_cpu_cross = 0.0;

    for (const auto& r : results) {
        if (r.dim != current_dim) {
            current_dim = r.dim;
            std::cout << "\n=== Dimension " << current_dim << " ===\n";
            std::cout << "   n | method            | statistic        | p-value   | stat_t [s]\n";
            std::cout << "---------------------------------------------------------------\n";
        }

        std::string method_name;
        switch (r.method) {
            case Method::CPU_BASELINE:    method_name = "CPU_BASELINE   "; break;
            case Method::CPU_CROSS_DIST:  method_name = "CPU_CROSS_DIST "; break;
            case Method::CUDA_CROSS_DIST: method_name = "CUDA_CROSS_DIST"; break;
            default:                      method_name = "UNKNOWN        "; break;
        }

        std::cout
            << std::setw(4) << r.n << " | "
            << method_name << " | "
            << std::setw(15) << std::setprecision(10) << r.statistic << " | "
            << std::setw(8) << r.p_value << " | "
            << std::fixed << std::setprecision(6) << r.stat_seconds
            << "\n";

        // =========================
        // Compare CUDA vs CPU_CROSS
        // =========================
        if (r.method == Method::CPU_CROSS_DIST) {
            last_cpu_cross = r.statistic;
        }

        if (r.method == Method::CUDA_CROSS_DIST) {
            double diff = std::abs(r.statistic - last_cpu_cross);
            std::cout << "       ↳ |CUDA - CPU_CROSS| = " << diff;

            if (diff < 1e-8) {
                std::cout << "  ✓ OK";
            } else {
                std::cout << "  ⚠ WARNING";
            }
            std::cout << "\n\n";
        }
    }

    std::cout << "\n------------\n";
    std::cout << "CUDA backend test finished\n";

    return 0;
}
