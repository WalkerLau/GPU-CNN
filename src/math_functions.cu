#include "math_functions.cuh"
#include <iostream>

#define CONV1 3*9*9*48	    // number of elements in filters of CONV1 is 3*9*9*48
#define CONV2 48*3*3*128	// number of elements in filters of CONV2 is 48*3*3*128
#define CONV3 128*3*3*128	// number of elements in filters of CONV3 is 128*3*3*128
#define CONV4 128*3*3*256	// number of elements in filters of CONV4 is 128*3*3*256
#define CONV5 256*3*3*192	// number of elements in filters of CONV5 is 256*3*3*192
#define CONV6 192*3*3*192	// number of elements in filters of CONV6 is 192*3*3*192
#define CONV7 192*3*3*128	// number of elements in filters of CONV7 is 192*3*3*128

__host__ void cuda_wrapper(float* A, float* B, float* C, const int n,
    const int m, const int k){
		cuda_matrix_procuct(A, B, C, n, m, k);
}

//allocate constant memory for filters
//assume that cuda_Grid has max{blocks} = 256, max{para_chn} = 4, max{fil_w} = 3
//__constant__ data_t filters[256*4*3*3];

// Match conv_layers
// matrix_procuct的输入分别为：输入数据首地址A、权重数据首地址B、输出数据首地址C、ofmap平面元素数量n、
// output volume的channel数m、一个filter的元素数量k。。。
__host__ void cuda_matrix_procuct(float* A, float* B, float* C, const int n,
	const int m, const int k) {

	switch(m*k){		
		case CONV1:{
			break;
		}			
		case CONV2:{
			break;
		}
		case CONV3:{
			break;
		}
		case CONV4:{
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 15;
			const int src_chn = 128;
			const int fil_w   = 3;
			const int dst_w   = 13;
			const int dst_chn = 256;
            
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);

			break;
		}
		case CONV5:{
			break;
		}
		case CONV6:{
			break;
		}
		case CONV7:{
			break;
		}
		default: 
			std::cout<<"Can't match conv_layers!"<<std::endl;

	}
}

__host__ void convolute(float* A, float* B, float* C,
	const int stride  ,
	const int src_w   ,
	const int src_chn ,
	const int fil_w   ,
	const int dst_w   ,
	const int dst_chn){
	
	//con_layer params
	const int src_h = src_w;
	const int fil_h = fil_w;
	const int dst_h = dst_w;
	
	//cuda_Grid params
	const int para_chn = 4;		//should also modify ifmaps size in conv_grid
	const int block_num = 256;
	dim3 dimGrid(1,1,block_num);
	dim3 dimBlock(fil_w, fil_h, para_chn);
	dim3 dimBlock_RA(src_w, src_h, para_chn);

	//create new space to store rearranged A
	data_t* ifmaps;
	cudaMalloc((void **)&ifmaps, sizeof(data_t)* block_num * para_chn * src_h * src_w);

	//create new space to store rearranged B
	data_t* filters;
	cudaMalloc((void **)&filters, sizeof(data_t) * block_num * para_chn * fil_h * fil_w);

	//initiate C to be zeros
	cudaMemset(C, 0, sizeof(float)*dst_chn*dst_h*dst_w);

	//get the whole ofmaps volume
	for(int ifm_lump = 0; ifm_lump < src_chn/para_chn; ++ifm_lump){
		
		//rearrange A to ifmaps
		rearrange_A<<<1,dimBlock_RA>>>(A, ifmaps, src_w, para_chn, ifm_lump, block_num);
		
		for(int ofm_lump = 0; ofm_lump < dst_chn/block_num; ++ofm_lump){

			//rearrange B to filters
			rearrange_B<<<dimGrid, dimBlock>>>(B, filters, src_chn, fil_w, ifm_lump, ofm_lump, para_chn, block_num);
			//cudaMemcpyToSymbol(filters, temp, sizeof(data_t) * block_num * para_chn * fil_h * fil_w);

			//get partial sum for block_num ofmaps
			conv_grid<<<dimGrid, dimBlock>>>
				(ifmaps, filters, C, src_w, fil_w, dst_w, ofm_lump, para_chn, block_num, stride);
			
		}
	}
	//free
	cudaFree(filters);
	cudaFree(ifmaps);
}

//rearrange A to ifmaps
__global__ void rearrange_A(float* A, data_t* ifmaps, 
	const int src_w,
	const int para_chn,
	const int ifm_lump,
	const int block_num
	){
	const int src_h = src_w;
	
	//grid index
	int bz = blockIdx.z;  int by = blockIdx.y;  int bx = blockIdx.x;
	int tz = threadIdx.z; int ty = threadIdx.y; int tx = threadIdx.x;
	int widx = bx*blockDim.x + tx;
	int hidx = by*blockDim.y + ty;
	int cidx = bz*blockDim.z + tz; 

	for(int i = 0, offset = 0; i < block_num; ++i){
		ifmaps[offset + cidx*src_h*src_w + hidx*src_w + widx] = 
			A[(ifm_lump*para_chn + cidx)*src_h*src_w + hidx*src_w + widx];
		__syncthreads();	
		offset += para_chn *  src_h * src_w;
	}

	//ifmaps[cidx*src_h*src_w + hidx*src_w + widx] = 
	//	A[(ifm_lump*para_chn + cidx)*src_h*src_w + hidx*src_w + widx];
	//	//__float2half(A[(ifm_lump*para_chn + cidx)*src_h*src_w + hidx*src_w + widx]);
	//__syncthreads();

}

//rearrange B to filters
__global__ void rearrange_B(float* B, data_t* filters,
	const int src_chn ,
	const int fil_w   ,
	const int ifm_lump,
	const int ofm_lump,
	const int para_chn,
	const int block_num
	){
	const int fil_h = fil_w;

	//int src_off = (ofm_lump*block_num*src_chn + ifm_lump*para_chn)*fil_h*fil_w;
	//for(int dst_off = 0, blk = 0; blk < block_num; ++blk){
	//	cudaMemcpyToSymbol(filters + dst_off, B + src_off, sizeof(float)*fil_h*fil_w*para_chn);
	//	dst_off += fil_h * fil_w * para_chn;
	//	src_off += src_chn * fil_h * fil_w;
	//}

	//int src_off = (ofm_lump*block_num*src_chn + ifm_lump*para_chn)*fil_h*fil_w;
	//for(int dst_off = 0, blk = 0; blk < block_num; ++blk){
	//	cudaMemcpy(temp + dst_off, B + src_off, sizeof(float)*fil_h*fil_w*para_chn, cudaMemcpyDeviceToDevice);
	//	dst_off += fil_h * fil_w * para_chn;
	//	src_off += src_chn * fil_h * fil_w;
	//}
	//
	//cudaMemcpyToSymbol(filters, temp, sizeof(float) * block_num * para_chn * fil_h * fil_w);

	//grid index
	int bz = blockIdx.z;  
	int tz = threadIdx.z; int ty = threadIdx.y; int tx = threadIdx.x;
	
	if( tx < fil_w && ty < fil_h && tz < para_chn){
		filters[ (bz * para_chn + tz) * fil_h * fil_w + ty * fil_w + tx] = 
			B[ (ofm_lump*block_num*src_chn + ifm_lump*para_chn + bz * src_chn + tz) * fil_h * fil_w + ty * fil_w + tx ];
			//__float2half(B[ (ofm_lump*block_num*src_chn + ifm_lump*para_chn + bz * src_chn + tz) * fil_h * fil_w + ty * fil_w + tx ]);
	}

	__syncthreads();

}

//get partial sum for block_num ofmaps
//only for stride = 1
__global__ void conv_grid(data_t* A, float* filters, float*C,
	const int src_w   ,
	const int fil_w   ,
	const int dst_w   ,
	const int ofm_lump,
	const int para_chn,
	const int block_num,
	const int stride
	){
	//con_layer params
	const int src_h = src_w;
	const int fil_h = fil_w;
	const int dst_h = dst_w;
 
	//grid index
	int bz = blockIdx.z;  //int by = blockIdx.y;  int bx = blockIdx.x;
	int tz = threadIdx.z; int ty = threadIdx.y; int tx = threadIdx.x;
	//int widx = bx*blockDim.x + tx;
	//int hidx = by*blockDim.y + ty;
	//int cidx = bz*blockDim.z + tz; 

	//allocate shared memory
	//[para_chn][fil_h][fil_w]
	__shared__ data_t ifmaps[4][3][3];
	__shared__ float  filtem[4][3][3];
	__shared__ float     res[4][3][3];

	//load filtem
	filtem[tz][ty][tx] = filters[(bz*para_chn + tz)*fil_h*fil_w + ty*fil_w + tx];
	__syncthreads();

	//sliding window
	for(int h = 0; h < dst_h; ++h){
		for(int w = 0; w < dst_w; ++w){
			//load ifmaps
			ifmaps[tz][ty][tx] = A[(bz*para_chn + tz)*src_h*src_w + (h*stride + ty)*src_w + (w*stride + tx)];
			__syncthreads();

			//calculate ofmap element
			res[tz][ty][tx] = ifmaps[tz][ty][tx] * filtem[tz][ty][tx];
			__syncthreads();
			if(0 == tx){
				for(int k = 1; k < fil_w; ++k){
					res[tz][ty][tx] += res[tz][ty][tx + k];
				}
			}	
			__syncthreads();
			if(0 == ty && 0 == tx){
				for(int k = 1; k < fil_h; ++k){
					res[tz][ty][tx] += res[tz][ty + k][tx];
				}
			}
			__syncthreads();
			if(0 == tz && 0 == ty && 0 == tx){
				for(int k = 1; k < para_chn; ++k){
					res[tz][ty][tx] += res[tz + k][ty][tx];
				}
			}
			__syncthreads();

			//store result to C
			if(0 == tz && 0 == ty && 0 == tx){
				C[(ofm_lump*block_num + bz)*dst_h*dst_w + h*dst_w + w] += res[tz][ty][tx];
			}
			__syncthreads();

		}
	}


	/*
	//allocate shared memory
	//[para_chn][src_h][src_w]
	__shared__ data_t ifmaps[4][15][15];
	__shared__ float  ofmaps[4][15][15];

	//A was re-arranged before conv_grid
	ifmaps[tz][ty][tx] = A[tz*src_h*src_w + ty*src_w + tx];
	__syncthreads();
	
	//calculate partial sum 
	//stride = 1
	for(int i = 0; i < fil_h; ++i){
		for(int j = 0; j < fil_w; ++j){
			if( (widx - j) >= 0 && (hidx - i) >= 0){
				//get dot product
				//filters was re-arranged as continuous
				ofmaps[tz][ty][tx] = 
								ifmaps[tz][ty][tx] *
								filters[cidx * fil_h * fil_w + ((hidx-i) % fil_h) * fil_w + (widx-j) % fil_w];
								//__half2float(ifmaps[tz][ty][tx]) * 
								//__half2float(filters[cidx * fil_h * fil_w + ((hidx-i) % fil_h) * fil_w + (widx-j) % fil_w]);
				__syncthreads();
				if( widx%fil_w == j){
					for(int k = 1; k < fil_w; ++k){
						ofmaps[tz][ty][tx] += ofmaps[tz][ty][tx + k];
					}
				}
				__syncthreads();
				if( hidx%fil_h == i){
					for(int k = 1; k < fil_h; ++k){
						ofmaps[tz][ty][tx] += ofmaps[tz][ty + k][tx];
					}
				}
				__syncthreads();
				if( cidx%blockDim.z == 0){
					for(int k = 1; k < blockDim.z; ++k){
						ofmaps[tz][ty][tx] += ofmaps[tz + k][ty][tx];
					}
				}
				__syncthreads();
				if( (cidx%blockDim.z) == 0 && (hidx%fil_h) == i && (widx%fil_w) == j){
					C[(ofm_lump*block_num + cidx/blockDim.z)*dst_h*dst_w + hidx*dst_w + widx] += ofmaps[tz][ty][tx];
				}
				__syncthreads();
			}
		}
	}
	*/


}
