/* 
 * This file is developed by Xuanzhi LIU (Walker LAU).
 * 
 * If you want to get the latest version of this project or met any problems,
 * please go to <https://github.com/WalkerLau/GPU-CNN> , 
 * I will try to help as much as I can.
 * 
 * You can redistribute this source codes and/or modify it under the terms of the BSD 2-Clause License.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 * 
 */

 #ifndef MATH_FUNCTIONS_CUH_
 #define MATH_FUNCTIONS_CUH_
 
 #include <cuda_runtime_api.h>
 #include <cuda_runtime.h>
 #include <stdio.h>
 #include <iostream>
 
 #include <cuda_fp16.h>
 typedef float data_t;
 
 #define CUDA_C 1
 #define CUDA_F 0
 
 __host__ void cuda_matrix_procuct(float* A, float* B, float* C, const int n,
     const int m, const int k);
 
 __host__ void convolute(float* A, float* B, float* C,
     const int stride  ,
     const int src_w   ,
     const int src_chn ,
     const int fil_w   ,
     const int dst_w   ,
     const int dst_chn);
 
 __global__ void conv_grid(data_t* A, float* B, float* C,
     const int src_w   ,
     const int src_chn ,
     const int fil_w   ,
     const int dst_w   ,
     const int ifm_lump,
     const int ofm_lump,
     const int block_num,
     const int stride,
     const int para_chn
     );
     
 __host__ void cuda_fc_wrapper(const float* A, const float* B, float* C, 
     const int vec_len, 
     const int dst_chn
     );
 
 __global__ void cuda_fc(const float* A, const float* B, float* C, 
     const int vec_len, 
     const int dst_chn
     );
 
 #endif