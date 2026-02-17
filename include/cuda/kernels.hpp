#pragma once
#include "Point.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define TILE_SIZE 32

namespace GPU {

template <int D, bool CompileTimeDim>
__global__ void compute_distance_matrix_tiled(
    const float* __restrict__ data,
    float* __restrict__ dist_mat,
    int total_points,
    int runtime_dim
);

// ------------------------------------------------------------
// P-value calculation (placeholder interface)
// ------------------------------------------------------------
//
// Contract:
// - dist_mat: pooled distance matrix D, flattened row-major, size N*N
// - total_points: N (= nX + nY)
// - nX: split index; [0..nX-1] are X, [nX..N-1] are Y
// - permutations: number of permutations (should include identity as p=0)
// - seed: RNG seed used on GPU for shuffling (implementation-defined)
//
// Returns: p-value as a single float.
//
// NOTE: Not implemented yet. The wrapper should call this after computing D.
float p_value_from_distance_matrix(
    const float* dist_mat,
    int total_points,
    int nX,
    int permutations,
    unsigned long long seed
);

// Host launcher for the distance matrix computation.
// This wraps the CUDA kernel launch so it can be called from normal C++.
void compute_distance_matrix(
    const float* d_data,
    float* d_dist_mat,
    int total_points,
    int dim
);




}


namespace GPU {
std::vector<std::vector<Point>> compute_distance_matrix(const std::vector<Point>& points);

}
