
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "Point.hpp"

namespace energy_test {

enum class Method : std::uint8_t {
	CPU_BASELINE = 0,
	CPU_CROSS_DIST = 1,
	CUDA_CROSS_DIST = 2,
};

struct BenchmarkConfig {
	std::size_t n = 100;
	std::size_t dim = 2;
	double delta = 0.2;
	int permutations = 199;
	std::uint32_t seed = 42;
	bool warmup = true;
};

struct BenchmarkResult {
	Method method;
	std::size_t n;
	std::size_t dim;
	double delta;
	int permutations;

	double statistic;
	double p_value;
	double stat_seconds;
	double p_value_seconds;
};

std::string method_name(Method m);

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg, Method method);

// Run a benchmark on explicitly provided samples.
// Contract:
// - X and Y must be non-empty
// - all points must have the same dimensionality
// - cfg.n/cfg.dim/cfg.delta are ignored (n and dim are derived from X/Y)
BenchmarkResult run_benchmark_xy(
	const BenchmarkConfig& cfg,
	Method method,
	const std::vector<Point>& X,
	const std::vector<Point>& Y
);

std::vector<BenchmarkResult> run_benchmarks(
	const std::vector<std::size_t>& sample_sizes,
	const std::vector<std::size_t>& dims,
	const BenchmarkConfig& base_cfg,
	const std::vector<Method>& methods
);

// Convenience: run multiple methods on one (X,Y) pair.
std::vector<BenchmarkResult> run_benchmarks_xy(
	const BenchmarkConfig& base_cfg,
	const std::vector<Method>& methods,
	const std::vector<Point>& X,
	const std::vector<Point>& Y
);

} // namespace energy_test

