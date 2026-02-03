#include "cpu_baseline.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

// Simple point type compatible with your API
struct Point {
    std::vector<double> coords;
    size_t size() const { return coords.size(); }
    double operator[](size_t i) const { return coords[i]; }
};

// Sample multivariate normal with diagonal covariance
std::vector<Point> sample_normal(
    size_t n,
    size_t dim,
    double mean_shift,
    std::mt19937 &gen
) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Point> out;
    out.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        Point p;
        p.coords.resize(dim);
        for (size_t d = 0; d < dim; ++d) {
            p.coords[d] = normal(gen) + mean_shift;
        }
        out.push_back(std::move(p));
    }
    return out;
}

using steady_clock = std::chrono::steady_clock;

void run_energy_test(size_t n, size_t dim) {
    static std::mt19937 gen(42); // fixed seed = reproducible

    constexpr double delta = 0.2;

    auto X = sample_normal(n, dim, 0.0, gen);
    auto Y = sample_normal(n, dim, delta, gen);

    // ---- Warmup ----
    CPU::energy_statistic(X, Y);

    // ---- Energy statistic timing ----
    auto t0 = steady_clock::now();
    double stat = CPU::energy_statistic(X, Y);
    auto t1 = steady_clock::now();

    // ---- P-value timing ----
    auto t2 = steady_clock::now();
    double pval = CPU::calculate_p_value(X, Y, /*permutations=*/199);
    auto t3 = steady_clock::now();

    std::chrono::duration<double> stat_time = t1 - t0;
    std::chrono::duration<double> pval_time = t3 - t2;

    std::cout
        << std::setw(6) << n
        << " | dim=" << dim
        << " | stat=" << std::setw(10) << stat
        << " | p=" << std::setw(8) << pval
        << " | stat_t=" << std::fixed << std::setprecision(4)
        << stat_time.count() << "s"
        << " | p_t=" << pval_time.count() << "s"
        << '\n';
}


int main() {
    std::vector<size_t> sample_sizes = {100, 500, 2000};
    std::vector<size_t> dims = {2, 3, 5};

    for (size_t dim : dims) {
        std::cout << "\n=== Dimension " << dim << " ===\n";
        for (size_t n : sample_sizes) {
            run_energy_test(n, dim);

        }
    }

    return 0;
}
