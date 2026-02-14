#include "kernel.hpp"
#include <cuda_runtime.h>
#include <cstdio>


__global__ void distance_kernel(const float* X, float* D, int num_points, int num_dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row 
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (i >= num_points || j >= num_points) return;

    float sum = 0.0f;

    #pragma unroll 100
    for (int k = 0; k < num_dims; k++) 
    { 
        if (k < num_dims) 
        {
            float diff = X[i * num_dims + k] - X[j * num_dims + k];
            sum += diff * diff;
        }
    }

    D[i * num_points + j] = sqrtf(sum);
}
