#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binder.hpp"

#ifdef ENERGY_TEST_HAS_CUDA
float energy_statistic_gpu(
    const std::vector<Point>& X,
    const std::vector<Point>& Y,
    int permutations,
    unsigned long long seed
);
#endif

namespace py = pybind11;

PYBIND11_MODULE(energy_py, m) {
    m.doc() = "Energy test benchmarks (CPU now, CUDA later)";

    using energy_test::BenchmarkConfig;
    using energy_test::BenchmarkResult;
    using energy_test::Method;
    using ::Point;

    py::class_<Point>(m, "Point")
        .def(py::init<>())
        .def_readwrite("coords", &Point::coords);

    py::enum_<Method>(m, "Method")
        .value("CPU_BASELINE", Method::CPU_BASELINE)
        .value("CPU_CROSS_DIST", Method::CPU_CROSS_DIST)
		.value("CUDA_CROSS_DIST", Method::CUDA_CROSS_DIST)
        .export_values();

    py::class_<BenchmarkConfig>(m, "BenchmarkConfig")
        .def(py::init<>())
        .def_readwrite("n", &BenchmarkConfig::n)
        .def_readwrite("dim", &BenchmarkConfig::dim)
        .def_readwrite("delta", &BenchmarkConfig::delta)
        .def_readwrite("permutations", &BenchmarkConfig::permutations)
        .def_readwrite("seed", &BenchmarkConfig::seed)
        .def_readwrite("warmup", &BenchmarkConfig::warmup);

    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def_readonly("method", &BenchmarkResult::method)
        .def_readonly("n", &BenchmarkResult::n)
        .def_readonly("dim", &BenchmarkResult::dim)
        .def_readonly("delta", &BenchmarkResult::delta)
        .def_readonly("permutations", &BenchmarkResult::permutations)
        .def_readonly("statistic", &BenchmarkResult::statistic)
        .def_readonly("p_value", &BenchmarkResult::p_value)
        .def_readonly("stat_seconds", &BenchmarkResult::stat_seconds)
        .def_readonly("p_value_seconds", &BenchmarkResult::p_value_seconds);

    m.def("method_name", &energy_test::method_name, py::arg("method"));
    m.def("run_benchmark", &energy_test::run_benchmark, py::arg("cfg"), py::arg("method"));
    m.def(
        "run_benchmark_xy",
        &energy_test::run_benchmark_xy,
        py::arg("cfg"),
        py::arg("method"),
        py::arg("X"),
        py::arg("Y")
    );
    m.def(
        "run_benchmarks",
        &energy_test::run_benchmarks,
        py::arg("sample_sizes"),
        py::arg("dims"),
        py::arg("base_cfg"),
        py::arg("methods")
    );
    m.def(
        "run_benchmarks_xy",
        &energy_test::run_benchmarks_xy,
        py::arg("base_cfg"),
        py::arg("methods"),
        py::arg("X"),
        py::arg("Y")
    );

    m.def(
        "energy_p_value_gpu",
        [](const std::vector<Point>& X, const std::vector<Point>& Y, int permutations, std::uint64_t seed) {
            #ifdef ENERGY_TEST_HAS_CUDA
            // Currently the GPU pipeline wrapper returns a p-value.
            return static_cast<double>(energy_statistic_gpu(X, Y, permutations, static_cast<unsigned long long>(seed)));
            #else
            throw std::runtime_error("CUDA not enabled in this build");
            #endif
        },
        py::arg("X"),
        py::arg("Y"),
        py::arg("permutations") = 199,
        py::arg("seed") = 42,
        "Compute energy-test p-value on GPU (distance matrix + permutation stats)."
    );
}
