#include "cuda/energy_cuda.hpp"
#include "cuda/kernels.cuh"
#include "common/point.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace energy_test {
namespace CUDA {

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + msg + ": " +
            cudaGetErrorString(err)
        );
    }
}

double energy_statistic(
    const std::vector<Point>& X,
    const std::vector<Point>& Y
) {
    // =========================
    // 0. Basic checks
    // =========================
    if (X.empty() || Y.empty()) {
        return 0.0;
    }
    if (X[0].coords.size() != 2 || Y[0].coords.size() != 2) {
        throw std::invalid_argument("CUDA backend supports only dim=2");
    }

    const int nX = static_cast<int>(X.size());
    const int nY = static_cast<int>(Y.size());

    // =========================
    // 1. Convert AoS -> SoA
    // =========================
    std::vector<double> h_xX(nX), h_yX(nX);
    std::vector<double> h_xY(nY), h_yY(nY);

    for (int i = 0; i < nX; ++i) {
        h_xX[i] = X[i].coords[0];
        h_yX[i] = X[i].coords[1];
    }
    for (int i = 0; i < nY; ++i) {
        h_xY[i] = Y[i].coords[0];
        h_yY[i] = Y[i].coords[1];
    }

    // =========================
    // 2. GPU allocation
    // =========================
    double *d_xX = nullptr, *d_yX = nullptr;
    double *d_xY = nullptr, *d_yY = nullptr;

    cuda_check(cudaMalloc(&d_xX, nX * sizeof(double)), "cudaMalloc d_xX");
    cuda_check(cudaMalloc(&d_yX, nX * sizeof(double)), "cudaMalloc d_yX");
    cuda_check(cudaMalloc(&d_xY, nY * sizeof(double)), "cudaMalloc d_xY");
    cuda_check(cudaMalloc(&d_yY, nY * sizeof(double)), "cudaMalloc d_yY");

    cuda_check(cudaMemcpy(d_xX, h_xX.data(), nX * sizeof(double),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy xX");
    cuda_check(cudaMemcpy(d_yX, h_yX.data(), nX * sizeof(double),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy yX");
    cuda_check(cudaMemcpy(d_xY, h_xY.data(), nY * sizeof(double),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy xY");
    cuda_check(cudaMemcpy(d_yY, h_yY.data(), nY * sizeof(double),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy yY");

    // =========================
    // 3. Kernel launch
    // =========================
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(
        (nX + TILE - 1) / TILE,
        (nY + TILE - 1) / TILE
    );

    const int numBlocks = grid.x * grid.y;

    double* d_partial = nullptr;
    cuda_check(cudaMalloc(&d_partial, numBlocks * sizeof(double)),
               "cudaMalloc d_partial");

    cross_distance_kernel<<<
        grid,
        block,
        TILE * TILE * sizeof(double)
    >>>(
        d_xX, d_yX,
        d_xY, d_yY,
        nX, nY,
        d_partial
    );

    cuda_check(cudaDeviceSynchronize(), "kernel execution");

    // =========================
    // 4. Reduction on CPU
    // =========================
    std::vector<double> h_partial(numBlocks);
    cuda_check(cudaMemcpy(h_partial.data(), d_partial,
                           numBlocks * sizeof(double),
                           cudaMemcpyDeviceToHost),
               "cudaMemcpy partial");

    double sumXY = 0.0;
    for (double v : h_partial) {
        sumXY += v;
    }

    // =========================
    // 5. Cleanup
    // =========================
    cudaFree(d_xX);
    cudaFree(d_yX);
    cudaFree(d_xY);
    cudaFree(d_yY);
    cudaFree(d_partial);

    return sumXY;
}

} // namespace CUDA
} // namespace energy_test
