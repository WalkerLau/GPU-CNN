#include "math_functions.cuh"

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
			//conv-net parameters
			//const int stride  = 4;
			//const int src_w   = 228;		
			//const int src_chn = 3;
			//const int fil_w   = 9;
			//const int dst_w   = 55;		
			//const int dst_chn = 48;
			//convolute(A, B, C,
			//		  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
			break;
		}			
		case CONV2:{
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 29;		
			const int src_chn = 48;
			const int fil_w   = 3;
			const int dst_w   = 27;		
			const int dst_chn = 128;
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
			break;
		}
		case CONV3:{
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 29;		
			const int src_chn = 128;
			const int fil_w   = 3;
			const int dst_w   = 27;		
			const int dst_chn = 128;
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
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
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 15;		
			const int src_chn = 256;
			const int fil_w   = 3;
			const int dst_w   = 13;		
			const int dst_chn = 192;
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
			break;
		}
		case CONV6:{
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 15;		
			const int src_chn = 192;
			const int fil_w   = 3;
			const int dst_w   = 13;		
			const int dst_chn = 192;
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
			break;
		}
		case CONV7:{
			//conv-net parameters
			const int stride  = 1;
			const int src_w   = 15;		
			const int src_chn = 192;
			const int fil_w   = 3;
			const int dst_w   = 13;		
			const int dst_chn = 128;
			convolute(A, B, C,
					  stride, src_w, src_chn, fil_w, dst_w, dst_chn);
			break;
		}
		default: 
			std::cout<<"ERROR! Can't match conv_layers!"<<std::endl;

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
	const int para_chn  = 1;		//should also modify memory allocation in conv_grid  
	const int block_num = dst_chn;
	dim3 dimGrid(1,1,block_num);
	dim3 dimBlock(dst_w, dst_h, 1);

	//initiate C to be zeros
	cudaMemset(C, 0, sizeof(float)*dst_chn*dst_h*dst_w);

	//get the whole ofmaps volume
	for(int ifm_lump = 0; ifm_lump < src_chn/para_chn; ++ifm_lump){
		for(int ofm_lump = 0; ofm_lump < dst_chn/block_num; ++ofm_lump){
			//get partial sum for block_num ofmaps
			conv_grid<<<dimGrid, dimBlock>>>
				(A, B, C, src_w, src_chn, fil_w, dst_w, ifm_lump, ofm_lump, block_num, stride, para_chn);
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
	const int stride,
	const int para_chn
	){
	//con_layer params
	const int src_h = src_w;
	const int fil_h = fil_w;
	const int dst_h = dst_w;
 
	//test time
	//clock_t clk_start, cnt = 0;
	//if(0 == bz && 0 == ty && 0 == tx) clk_start = clock();
	//if(0 == bz && 0 == ty && 0 == tx) printf("....calculate = %ld\n",clock() - clk_start);

	//grid index
	int bz  = blockIdx.z;  
	int ty  = threadIdx.y;	int tx = threadIdx.x;
	int tid = ty*blockDim.x + tx;

	//allocate shared memory & registers
	__shared__ data_t ifmaps[1*29*29];	//[para_chn * src_h * src_w]
	float filters[1*3*3]; 	//[para_chn * fil_h * fil_w]
	float res = 0;

	//load ifmaps
	for(int i = 0; i*dst_h*dst_w < para_chn*src_h*src_w; ++i){
		if(i*dst_h*dst_w + tid < para_chn*src_h*src_w){
			ifmaps[i*dst_h*dst_w + tid] = A[ifm_lump*para_chn*src_h*src_w + i*dst_h*dst_w + tid];
		}
	}

	//load filters
	for(int i = 0; i < para_chn*fil_h*fil_w; ++i){
		filters[i] = B[(ofm_lump*block_num*src_chn + ifm_lump*para_chn + bz*src_chn)*fil_h*fil_w + i];
	}

	//calculate partial sum
	for(int c = 0, k = 0; c < para_chn; ++c){
		for(int h = 0; h < fil_h; ++h){
			for(int w = 0; w < fil_w; ++w){
				res += ifmaps[c*src_h*src_w + ty*stride*src_w + tx*stride + h*src_w + w] * filters[k];
				++k; 
			}
		}
	}

	//write C
	C[ofm_lump*block_num*dst_h*dst_w + bz*dst_h*dst_w + ty*dst_w + tx] += res;

}


__host__ void cuda_fc_wrapper(const float* A, const float* B, float* C, 
	const int vec_len, 
	const int dst_chn
	){
	//copy data to GPU global memory
	float* src;
	float* fil;
	float* dst;
	cudaMalloc((void **)&src, vec_len * sizeof(float));
	cudaMalloc((void **)&fil, vec_len * dst_chn * sizeof(float));
	cudaMalloc((void **)&dst, dst_chn * sizeof(float));
	cudaMemcpy(src, A, vec_len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fil, B, vec_len * dst_chn * sizeof(float), cudaMemcpyHostToDevice);

	//configure fc info & calculate
	if(4096 == dst_chn){
		cuda_fc<<<4,1024>>>(src, fil, dst, vec_len, dst_chn);
	}
	else if(2048 == dst_chn){
		cuda_fc<<<2,1024>>>(src, fil, dst, vec_len, dst_chn);
	}
	else{
		std::cout<<"ERROR! Cannot match FC layer!"<<std::endl;
	}

	//copy result to host
	cudaMemcpy(C, dst, dst_chn * sizeof(float), cudaMemcpyDeviceToHost);
	//free Malloc
	cudaFree(dst);
	cudaFree(fil);
	cudaFree(src);

}

__global__ void cuda_fc(const float* A, const float* B, float* C, 
	const int vec_len, 
	const int dst_chn
	){
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int tid = bx*blockDim.x + tx;
	
	//allocate memory
	__shared__ float src[128*6*6];		//max src size

	//load src
	for(int i = 0; i*blockDim.x < vec_len; ++i){
		if(i*blockDim.x + tx < vec_len){
			src[i*blockDim.x + tx] = A[i*blockDim.x + tx];
		}
	}

	//calculate res
	float res = 0;
	for(int i = 0; i < vec_len; ++i){
		res += src[i] * B[tid*vec_len + i];
	}

	//write C
	C[tid] = res;
}