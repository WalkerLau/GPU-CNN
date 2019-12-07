#include "math_functions.cuh"
#include <iostream>

#define CONV1 3*9*9*48	    // number of elements in filters of CONV1 is 3*9*9*48
#define CONV2 48*3*3*128	// number of elements in filters of CONV2 is 48*3*3*128
#define CONV3 128*3*3*128	// number of elements in filters of CONV3 is 128*3*3*128
#define CONV4 128*3*3*256	// number of elements in filters of CONV4 is 128*3*3*256
#define CONV5 256*3*3*192	// number of elements in filters of CONV5 is 256*3*3*192
#define CONV6 192*3*3*192	// number of elements in filters of CONV6 is 192*3*3*192
#define CONV7 192*3*3*128	// number of elements in filters of CONV7 is 192*3*3*128

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
	const int dst_h = dst_w;
	
	//cuda_Grid params
	const int block_num = 256;
	dim3 dimGrid(1,1,block_num);
	dim3 dimBlock(dst_w, dst_h, 1);

	//initiate C to be zeros
	cudaMemset(C, 0, sizeof(float)*dst_chn*dst_h*dst_w);

	//get the whole ofmaps volume
	for(int ifm_lump = 0; ifm_lump < src_chn; ++ifm_lump){
		for(int ofm_lump = 0; ofm_lump < dst_chn/block_num; ++ofm_lump){
			//get partial sum for block_num ofmaps
			conv_grid<<<dimGrid, dimBlock>>>
				(A, B, C, src_w, src_chn, fil_w, dst_w, ifm_lump, ofm_lump, block_num, stride);
		}
	}
}

//get partial sum for block_num ofmaps
__global__ void conv_grid(data_t* A, float* B, float*C,
	const int src_w   ,
	const int src_chn ,
	const int fil_w   ,
	const int dst_w   ,
	const int ifm_lump,
	const int ofm_lump,
	const int block_num,
	const int stride
	){
	//con_layer params
	const int src_h = src_w;
	const int fil_h = fil_w;
	const int dst_h = dst_w;
 
	//test time
	//clock_t clk_start, cnt = 0;

	//grid index
	int bz  = blockIdx.z;  
	int ty  = threadIdx.y;	int tx = threadIdx.x;
	int tid = ty*blockDim.x + tx;

	//allocate shared memory & registers
	__shared__ data_t ifmaps[15*15];	//[src_h * src_w]
	float filters[3*3]; 	//[fil_h * fil_w]
	float res = 0;

	//load ifmaps
	for(int i = 0; i*dst_h*dst_w < src_h*src_w; ++i){
		if(i*dst_h*dst_w + tid < src_h*src_w){
			ifmaps[i*dst_h*dst_w + tid] = A[ifm_lump*src_h*src_w + i*dst_h*dst_w + tid];
		}
		__syncthreads();
	}

	//load filters
	for(int i = 0; i < fil_h*fil_w; ++i){
		filters[i] = B[(ofm_lump*block_num*src_chn + ifm_lump + bz*src_chn)*fil_h*fil_w + i];
	}

	//calculate partial sum
	for(int i = 0, k = 0; i < fil_h; ++i){
		for(int j = 0; j < fil_w; ++j){
			res += ifmaps[ty*stride*src_w + tx*stride + i*src_w + j] * filters[k];
			++k; 
		}
	}

	//write C
	C[ofm_lump*block_num*dst_h*dst_w + bz*dst_h*dst_w + ty*dst_w + tx] += res;

}
