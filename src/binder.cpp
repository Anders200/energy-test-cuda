
#include "binder.hpp"

#include "cpu/cpu_baseline.hpp"
#include "cpu/cpu_cross_dist.hpp"

#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>

namespace energy_test {
namespace {

struct Point {
	std::vector<double> coords;
	std::size_t size() const { return coords.size(); }
	double operator[](std::size_t i) const { return coords[i]; }
};

using steady_clock = std::chrono::steady_clock;

std::vector<Point> sample_normal(
	std::size_t n,
	std::size_t dim,
	double mean_shift,
	std::mt19937& gen
) {
	std::normal_distribution<double> normal(0.0, 1.0);
	std::vector<Point> out;
	out.reserve(n);

	for (std::size_t i = 0; i < n; ++i) {
		Point p;
		p.coords.resize(dim);
		for (std::size_t d = 0; d < dim; ++d) {
			p.coords[d] = normal(gen) + mean_shift;
		}
		out.push_back(std::move(p));
	}
	return out;
}

} // namespace

std::string method_name(Method m) {
	switch (m) {
		case Method::CPU_BASELINE: return "cpu_baseline";
		case Method::CPU_CROSS_DIST: return "cpu_cross_dist";
		default: return "unknown";
	}
}

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg, Method method) {
	std::mt19937 gen(cfg.seed);

	auto X = sample_normal(cfg.n, cfg.dim, 0.0, gen);
	auto Y = sample_normal(cfg.n, cfg.dim, cfg.delta, gen);

	auto run_stat = [&]() {
		switch (method) {
			case Method::CPU_BASELINE:
				return CPU::energy_statistic(X, Y);
			case Method::CPU_CROSS_DIST:
				return CPU_cross_dist::energy_statistic(X, Y);
			default:
				throw std::invalid_argument("Unknown method");
		}
	};

	auto run_pval = [&]() {
		switch (method) {
			case Method::CPU_BASELINE:
				return CPU::calculate_p_value(X, Y, cfg.permutations);
			case Method::CPU_CROSS_DIST:
				return CPU_cross_dist::calculate_p_value(X, Y, cfg.permutations);
			default:
				throw std::invalid_argument("Unknown method");
		}
	};

	if (cfg.warmup) {
		(void)run_stat();
	}

	auto t0 = steady_clock::now();
	double stat = run_stat();
	auto t1 = steady_clock::now();

	auto t2 = steady_clock::now();
	double pval = run_pval();
	auto t3 = steady_clock::now();

	std::chrono::duration<double> stat_time = t1 - t0;
	std::chrono::duration<double> pval_time = t3 - t2;

	BenchmarkResult r{};
	r.method = method;
	r.n = cfg.n;
	r.dim = cfg.dim;
	r.delta = cfg.delta;
	r.permutations = cfg.permutations;
	r.statistic = stat;
	r.p_value = pval;
	r.stat_seconds = stat_time.count();
	r.p_value_seconds = pval_time.count();
	return r;
}

std::vector<BenchmarkResult> run_benchmarks(
	const std::vector<std::size_t>& sample_sizes,
	const std::vector<std::size_t>& dims,
	const BenchmarkConfig& base_cfg,
	const std::vector<Method>& methods
) {
	std::vector<BenchmarkResult> out;
	for (auto dim : dims) {
		for (auto n : sample_sizes) {
			for (auto m : methods) {
				BenchmarkConfig cfg = base_cfg;
				cfg.n = n;
				cfg.dim = dim;
				out.push_back(run_benchmark(cfg, m));
			}
		}
	}
	return out;
}

} // namespace energy_test

