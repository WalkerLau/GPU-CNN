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
//assume that cuda_Grid has 16 blocks while max{para_chn} = 4, max{fil_w} = 9
__constant__ float filters[16*4*9*9];

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
	const int para_chn = 1;
	const int block_num = 1;
	dim3 dimGrid(1,1,block_num);
	dim3 dimBlock(src_w, src_h, para_chn);

	//create new space to store rearranged A
	float* ifmaps;
	cudaMalloc((void **)&ifmaps, sizeof(float)*para_chn*src_h*src_w);

	//create temp space to store rearranged B
	float* temp;
	cudaMalloc((void **)&temp, sizeof(float) * block_num * para_chn * fil_h * fil_w);

	//initiate C to be zeros
	cudaMemset(C, 0, sizeof(float)*dst_chn*dst_h*dst_w);

	//get the whole ofmaps volume
	for(int ifm_lump = 0; ifm_lump < src_chn/para_chn; ++ifm_lump){
		
		//rearrange A to ifmaps
		rearrange_A<<<1,dimBlock>>>(A, ifmaps, src_w, para_chn, ifm_lump);
		
		for(int ofm_lump = 0; ofm_lump < dst_chn/block_num; ++ofm_lump){

			//rearrange B to filters
			rearrange_B(B, temp, src_chn, fil_w, ifm_lump, ofm_lump, para_chn, block_num);
			cudaDeviceSynchronize();

			//get partial sum for block_num ofmaps
			conv_grid<<<dimGrid, dimBlock>>>
				(ifmaps, C, src_w, fil_w, dst_w, ofm_lump, para_chn, block_num);
			cudaDeviceSynchronize();
			
		}
	}
	//free
	cudaFree(temp);
	cudaFree(ifmaps);
}

//rearrange A to ifmaps
__global__ void rearrange_A(float* A, float* ifmaps, 
	const int src_w,
	const int para_chn,
	const int ifm_lump
	){
	const int src_h = src_w;
	
	//grid index
	int bz = blockIdx.z;  int by = blockIdx.y;  int bx = blockIdx.x;
	int tz = threadIdx.z; int ty = threadIdx.y; int tx = threadIdx.x;
	int widx = bx*blockDim.x + tx;
	int hidx = by*blockDim.y + ty;
	int cidx = bz*blockDim.z + tz; 
	ifmaps[cidx*src_h*src_w + hidx*src_w + widx] = 
			A[(ifm_lump*para_chn + cidx)*src_h*src_w + hidx*src_w + widx];
	__syncthreads();

}

//rearrange B to filters
__host__ void rearrange_B(float* B, float* temp,
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
	
	int src_off = (ofm_lump*block_num*src_chn + ifm_lump*para_chn)*fil_h*fil_w;
	for(int dst_off = 0, blk = 0; blk < block_num; ++blk){
		cudaMemcpy(temp + dst_off, B + src_off, sizeof(float)*fil_h*fil_w*para_chn, cudaMemcpyDeviceToDevice);
		dst_off += fil_h * fil_w * para_chn;
		src_off += src_chn * fil_h * fil_w;
	}

	cudaMemcpyToSymbol(filters, temp, sizeof(float) * block_num * para_chn * fil_h * fil_w);

}

//get partial sum for block_num ofmaps
//only for stride = 1
__global__ void conv_grid(float* A, float*C,
	const int src_w   ,
	const int fil_w   ,
	const int dst_w   ,
	const int ofm_lump,
	const int para_chn,
	const int block_num
	){
	//con_layer params
	const int src_h = src_w;
	const int fil_h = fil_w;
	const int dst_h = dst_w;
 
	//grid index
	int bz = blockIdx.z;  int by = blockIdx.y;  int bx = blockIdx.x;
	int tz = threadIdx.z; int ty = threadIdx.y; int tx = threadIdx.x;
	int widx = bx*blockDim.x + tx;
	int hidx = by*blockDim.y + ty;
	int cidx = bz*blockDim.z + tz; 

	//allocate shared memory
	//[para_chn][src_h][src_w]
	__shared__ float ifmaps[1][15][15];
	__shared__ float ofmaps[1][15][15];

	//A was re-arranged before conv_grid
	ifmaps[cidx][hidx][widx] = A[tz*src_h*src_w + ty*src_w + tx];
	__syncthreads();

	//calculate partial sum 
	//stride = 1
	for(int i = 0; i < fil_h; ++i){
		for(int j = 0; j < fil_w; ++j){
			if( (widx - j) >= 0 && (hidx - i) >= 0){
				//filters was re-arranged as continuous
				ofmaps[cidx][hidx][widx] = ifmaps[cidx][hidx][widx] * 
										   filters[cidx * fil_h * fil_w + ((hidx-i) % fil_h) * fil_w + (widx-j) % fil_w];
				__syncthreads();
				if( widx%fil_w == j){
					for(int k = 1; k < fil_w; ++k){
						ofmaps[cidx][hidx][widx] += ofmaps[cidx][hidx][widx+k];
					}
				}
				__syncthreads();
				if( hidx%fil_h == i){
					for(int k = 1; k < fil_h; ++k){
						ofmaps[cidx][hidx][widx] += ofmaps[cidx][hidx+k][widx];
					}
				}
				__syncthreads();
				if( cidx%blockDim.z == 0){
					for(int k = 1; k < blockDim.z; ++k){
						ofmaps[cidx][hidx][widx] += ofmaps[cidx+k][hidx][widx];
					}
				}
				__syncthreads();
				if( (cidx%blockDim.z) == 0 && (hidx%fil_h) == i && (widx%fil_w) == j){
					C[(ofm_lump*block_num + cidx/blockDim.z)*dst_h*dst_w + hidx*dst_w + widx] += ofmaps[cidx][hidx][widx];
				}
				__syncthreads();
			}
		}
	}

}
