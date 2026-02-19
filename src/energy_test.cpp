#include "cuda/kernels.hpp"

#include "Point.hpp"

#include <stdexcept>

namespace {
static void cuda_check(cudaError_t err, const char* what) {
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA error: ") + what + ": " + cudaGetErrorString(err));
	}
}
} // namespace

float energy_statistic_gpu(
    const std::vector<Point>& X,
    const std::vector<Point>& Y,
    int permutations,
    unsigned long long seed
) {
    int n = static_cast<int>(X.size());
    int m = static_cast<int>(Y.size());
    int dim = static_cast<int>(X[0].coords.size());

    // matrix
    /*
    ctemplate <int D, bool CompileTimeDim>
__global__ void compute_distance_matrix_tiled(
    const float* __restrict__ data,
    float* __restrict__ dist_mat,
    int total_points,
    int runtime_dim
);
*/
    const int N = n + m;
    if (N <= 0) return 0.0f;
    if (dim <= 0) throw std::invalid_argument("dim must be > 0");

    // Flatten pooled data = [X... Y...] as AoS float buffer
    std::vector<float> h_data(static_cast<size_t>(N) * static_cast<size_t>(dim));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < dim; ++d) {
            if (i < n) {
                h_data[static_cast<size_t>(i) * dim + d] = static_cast<float>(X[i].coords[d]);
            } else {
                h_data[static_cast<size_t>(i) * dim + d] = static_cast<float>(Y[i - n].coords[d]);
            }
        }
    }

    float* d_data = nullptr;
    float* d_D = nullptr;
    const size_t data_bytes = h_data.size() * sizeof(float);
    const size_t D_bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    cuda_check(cudaMalloc(&d_data, data_bytes), "cudaMalloc d_data");
    cuda_check(cudaMalloc(&d_D, D_bytes), "cudaMalloc d_D");
    cuda_check(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D data");

    // Launch distance matrix kernel
    GPU::compute_distance_matrix(d_data, d_D, N, dim);

    // P-value computation based on pooled distance matrix D
    // (simple implementation for now; can be improved later).
    float p = GPU::p_value_from_distance_matrix(d_D, N, n, permutations, seed);

    cudaFree(d_D);
    cudaFree(d_data);
    return p;
}