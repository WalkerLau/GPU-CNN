#ifndef MATH_FUNCTIONS_CUH_
#define MATH_FUNCTIONS_CUH_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ void cuda_matrix_procuct(float* A, float* B, float* C, const int n,
    const int m, const int k);

__host__ void convolute(float* A, float* B, float* C,
    const int stride  ,
    const int src_w   ,
    const int src_chn ,
    const int fil_w   ,
    const int dst_w   ,
    const int dst_chn);

__global__ void rearrange_A(float* A, float* ifmaps, 
    const int src_w,
    const int para_chn,
    const int ifm_lump
    );

__global__ void rearrange_B(float* B, float* filters,
    const int src_chn ,
    const int fil_w   ,
    const int ifm_lump,
    const int ofm_lump,
    const int para_chn,
    const int block_num
    );

__global__ void conv_grid(float* A, float* filters, float*C,
    const int src_w   ,
    const int fil_w   ,
    const int dst_w   ,
    const int ofm_lump,
    const int para_chn,
    const int block_num
    );
    

#endif