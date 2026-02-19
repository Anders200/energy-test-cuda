#include "cuda/kernels.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + what + ": " + cudaGetErrorString(err));
    }
}

struct Triple {
    float xx;
    float yy;
    float xy;
};

__global__ void energy_stat_partials_from_pooled_D_kernel(
    const float* __restrict__ D,
    int N,
    int nX,
    const int* __restrict__ perm, // length N or nullptr for identity
    Triple* __restrict__ out_partials // length gridDim.x
) {
    extern __shared__ float s[];
    float* sXX = s;
    float* sYY = s + blockDim.x;
    float* sXY = s + 2 * blockDim.x;

    const int tid = threadIdx.x;
    const int block = blockIdx.x;
    const int strideBlocks = gridDim.x;

    float localXX = 0.0f;
    float localYY = 0.0f;
    float localXY = 0.0f;

    for (int ii = block; ii < N; ii += strideBlocks) {
        if (ii < nX) {
            const int pi = perm ? perm[ii] : ii;
            for (int jj = ii + 1 + tid; jj < nX; jj += blockDim.x) {
                const int pj = perm ? perm[jj] : jj;
                localXX += D[pi * N + pj];
            }
            for (int jj = nX + tid; jj < N; jj += blockDim.x) {
                const int pj = perm ? perm[jj] : jj;
                localXY += D[pi * N + pj];
            }
        } else {
            const int pi = perm ? perm[ii] : ii;
            for (int jj = ii + 1 + tid; jj < N; jj += blockDim.x) {
                const int pj = perm ? perm[jj] : jj;
                localYY += D[pi * N + pj];
            }
        }
    }

    sXX[tid] = localXX;
    sYY[tid] = localYY;
    sXY[tid] = localXY;
    __syncthreads();

    for (int sStride = blockDim.x / 2; sStride > 0; sStride >>= 1) {
        if (tid < sStride) {
            sXX[tid] += sXX[tid + sStride];
            sYY[tid] += sYY[tid + sStride];
            sXY[tid] += sXY[tid + sStride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_partials[block] = Triple{ sXX[0], sYY[0], sXY[0] };
    }
}

__global__ void reduce_triples_kernel(const Triple* __restrict__ in, int n, Triple* __restrict__ out) {
    extern __shared__ float s[];
    float* sXX = s;
    float* sYY = s + blockDim.x;
    float* sXY = s + 2 * blockDim.x;

    const int tid = threadIdx.x;
    float xx = 0.0f, yy = 0.0f, xy = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const Triple t = in[i];
        xx += t.xx;
        yy += t.yy;
        xy += t.xy;
    }
    sXX[tid] = xx;
    sYY[tid] = yy;
    sXY[tid] = xy;
    __syncthreads();

    for (int sStride = blockDim.x / 2; sStride > 0; sStride >>= 1) {
        if (tid < sStride) {
            sXX[tid] += sXX[tid + sStride];
            sYY[tid] += sYY[tid + sStride];
            sXY[tid] += sXY[tid + sStride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = Triple{ sXX[0], sYY[0], sXY[0] };
    }
}

static float energy_stat_from_D_on_gpu(const float* d_D, int N, int nX, const int* d_perm, int blocks, int threads) {
    if (blocks < 1) blocks = 1;

    Triple* d_partials = nullptr;
    Triple* d_total = nullptr;
    cuda_check(cudaMalloc(&d_partials, static_cast<size_t>(blocks) * sizeof(Triple)), "cudaMalloc d_partials");
    cuda_check(cudaMalloc(&d_total, sizeof(Triple)), "cudaMalloc d_total");

    const size_t shmem = 3ull * static_cast<size_t>(threads) * sizeof(float);
    energy_stat_partials_from_pooled_D_kernel<<<blocks, threads, shmem>>>(d_D, N, nX, d_perm, d_partials);
    cuda_check(cudaGetLastError(), "energy_stat_partials launch");

    reduce_triples_kernel<<<1, threads, shmem>>>(d_partials, blocks, d_total);
    cuda_check(cudaGetLastError(), "reduce_triples launch");
    cuda_check(cudaDeviceSynchronize(), "energy_stat reduce sync");

    Triple h;
    cuda_check(cudaMemcpy(&h, d_total, sizeof(Triple), cudaMemcpyDeviceToHost), "copy total triple");

    cudaFree(d_total);
    cudaFree(d_partials);

    const float nY = float(N - nX);
    const float fnX = float(nX);
    return (2.0f / (fnX * nY)) * h.xy
        - (2.0f / (fnX * fnX)) * h.xx
        - (2.0f / (nY * nY)) * h.yy;
}

__global__ void init_identity_perm_kernel(int* perm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) perm[i] = i;
}

__global__ void fisher_yates_shuffle_inplace_kernel(int* perm, int N, unsigned long long seed) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, /*subsequence=*/0, /*offset=*/0, &st);
    for (int i = N - 1; i > 0; --i) {
        unsigned int r = curand(&st);
        int j = (int)(r % (unsigned int)(i + 1));
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}
} // namespace

namespace GPU {

__device__ __forceinline__ void triangular_tile_index_to_ij(int t, int tiles, int& out_i, int& out_j) {
    // Map t in [0, tiles*(tiles+1)/2) to (i,j) with 0<=i<=j<tiles.
    // lin search becuase tiles is usually small, sqrt would be complex
    int acc = 0;
    for (int i = 0; i < tiles; ++i) {
        const int rowLen = tiles - i;
        if (t < acc + rowLen) {
            out_i = i;
            out_j = i + (t - acc);
            return;
        }
        acc += rowLen;
    }
    out_i = 0;
    out_j = 0;
}

template <int D, bool CompileTimeDim>
__global__ void compute_distance_matrix_tiled
(
    const float* __restrict__ data,
    float* __restrict__ dist_mat,
    int total_points,
    int runtime_dim
)
{
    extern __shared__ float s_mem[];
    float* tile_row = s_mem; 
    float* tile_col = &s_mem[TILE_SIZE * (CompileTimeDim ? D : runtime_dim)];

    int dim = CompileTimeDim ? D : runtime_dim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	const int tiles = (total_points + TILE_SIZE - 1) / TILE_SIZE;
	int tile_i = 0;
	int tile_j = 0;
	triangular_tile_index_to_ij((int)blockIdx.x, tiles, tile_i, tile_j);

    int row = tile_i * TILE_SIZE + ty;
    int col = tile_j * TILE_SIZE + tx;

    float dist = 0.0f;

    if (row < total_points && col < total_points) 
    {
            
        for (int d = 0; d < dim; ++d) 
        {
            if (ty < TILE_SIZE) tile_row[ty * dim + d] = data[row * dim + d];
            if (tx < TILE_SIZE) tile_col[tx * dim + d] = data[col * dim + d];
        }
        __syncthreads();

        float diff_sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < dim; ++d) 
        {
            float a = tile_row[ty * dim + d];
            float b = tile_col[tx * dim + d];
            float diff = a - b;
            diff_sum += diff * diff;
        }
        dist = sqrtf(diff_sum);
        
        dist_mat[row * total_points + col] = dist;
        if (row != col) {
            dist_mat[col * total_points + row] = dist;
        }
    }
}

} // namespace GPU

void GPU::compute_distance_matrix(
    const float* d_data,
    float* d_dist_mat,
    int total_points,
    int dim
) {
    if (total_points <= 0 || dim <= 0) return;
    dim3 block(TILE_SIZE, TILE_SIZE);
	const int tiles = (total_points + TILE_SIZE - 1) / TILE_SIZE;
	const int triTiles = tiles * (tiles + 1) / 2;
    dim3 grid(triTiles);
    const size_t shared_bytes = 2ull * TILE_SIZE * static_cast<size_t>(dim) * sizeof(float);
    GPU::compute_distance_matrix_tiled<0, false><<<grid, block, shared_bytes>>>(
        d_data, d_dist_mat, total_points, dim
    );
    cuda_check(cudaGetLastError(), "compute_distance_matrix_tiled launch");
    cuda_check(cudaDeviceSynchronize(), "compute_distance_matrix_tiled sync");
}


float GPU::p_value_from_distance_matrix(
    const float* dist_mat,
    int total_points,
    int nX,
    int permutations,
    unsigned long long seed
) {
    if (total_points <= 0) return 1.0f;
    if (nX <= 0 || nX >= total_points) return 1.0f;
    if (permutations <= 0) permutations = 1;

    const int threads = 256;
    int blocks = (total_points + 7) / 8;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;

    const float observed = energy_stat_from_D_on_gpu(dist_mat, total_points, nX, nullptr, blocks, threads);

    int* d_perm = nullptr;
    cuda_check(cudaMalloc(&d_perm, static_cast<size_t>(total_points) * sizeof(int)), "cudaMalloc d_perm");
    std::vector<float> perm_stats(static_cast<size_t>(permutations));

    perm_stats[0] = observed;

    for (int p = 1; p < permutations; ++p) {
        const int t = 256;
        const int b = (total_points + t - 1) / t;
        init_identity_perm_kernel<<<b, t>>>(d_perm, total_points);
        cuda_check(cudaGetLastError(), "init_identity_perm launch");
        fisher_yates_shuffle_inplace_kernel<<<1, 1>>>(d_perm, total_points, seed ^ (unsigned long long)p * 0xD2B74407B1CE6E93ull);
        cuda_check(cudaGetLastError(), "fisher_yates_shuffle launch");

        perm_stats[static_cast<size_t>(p)] = energy_stat_from_D_on_gpu(dist_mat, total_points, nX, d_perm, blocks, threads);
    }

    cudaFree(d_perm);

    int count = 0;
    for (float s : perm_stats) {
        if (s >= observed) ++count;
    }
    // +1 smoothing (same as CPU)
    return float(count + 1) / float(permutations + 1);
}
