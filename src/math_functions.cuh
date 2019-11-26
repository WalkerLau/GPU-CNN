#ifndef MATH_FUNCTIONS_CUH_
#define MATH_FUNCTIONS_CUH_


__global__ void cuda_matrix_procuct(float* A, float* B, float* C, const int n,
    const int m, const int k, bool ta = false, bool tb = false);

#endif