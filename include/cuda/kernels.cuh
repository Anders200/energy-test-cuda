#pragma once

__global__
void cross_distance_kernel(
    const double* xX, const double* yX,
    const double* xY, const double* yY,
    int nX, int nY,
    double* partial_sums
);
