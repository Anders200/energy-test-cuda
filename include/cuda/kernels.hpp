#pragma once
#include "Point.hpp"

namespace CUDA {

// kernel for filling distance matrix 
void compute_distance_matrix(const float* X, float* D, int num_points, int num_dims);
// kernel for computing energy statistic from distance matrices




}
