#include "math_functions.cuh"

#define CONV1 3*9*9*48	    // number of elements in filters of CONV1 is 3*9*9*48
#define CONV2 48*3*3*128	// number of elements in filters of CONV2 is 48*3*3*128
#define CONV3 128*3*3*128	// number of elements in filters of CONV3 is 128*3*3*128
#define CONV4 128*3*3*256	// number of elements in filters of CONV4 is 128*3*3*256
#define CONV5 256*3*3*192	// number of elements in filters of CONV5 is 256*3*3*192
#define CONV6 192*3*3*192	// number of elements in filters of CONV6 is 192*3*3*192
#define CONV7 192*3*3*128	// number of elements in filters of CONV7 is 192*3*3*128

void cuda_wrapper(float* A, float* B, float* C, const int n,
    const int m, const int k, bool ta = false, bool tb = false){
		cuda_matrix_procuct<<<1,1>>>(A, B, C, n, m, k, ta, tb);
	}

// 计算一个ofmap元素
// x为filter地址，y为ifmaps地址
__device__ float dev_dot(float* x, float* y, const long& k){	
	float inner_prod = 0;
	for(int i = 0; i < k; i++){
		inner_prod += x[i]*y[i];
	}
	return inner_prod;
}

// matrix_procuct的输入分别为：输入数据首地址A、权重数据首地址B、输出数据首地址C、ofmap平面元素数量n、
// output volume的channel数m、一个filter的元素数量k。。。
__global__ void cuda_matrix_procuct(float* A, float* B, float* C, const int n,
	const int m, const int k, bool ta, bool tb) {

	float* filter = B;
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
			const int stride = 1;
			const int src_w = 15;
			const int src_channels = 128;
			const int kernel_w = 3;
			const int dst_w = 13;
			const int src_h = src_w;
			const int kernel_h = kernel_w;
			const int dst_h = dst_w;
			//const int end_h = src_h - kernel_h + 1;
  			//const int end_w = src_w - kernel_w + 1;
            
            // allocate ifmaps memory
            float ifmaps[kernel_w * kernel_h * src_channels];

			// 循环完毕将产生完整的output volume
			for (int i = 0, idx = 0; i < m; ++i) {	
				// 乘加运算：执行一次该循环将产生一个ofmap元素，循环完毕将产生output volume中的一张ofmap
				for (int j = 0; j < dst_h; ++j) {	//纵向移窗
					for(int l = 0; l < dst_w; ++l, ++idx){  //横向移窗
						//load ifmap
						//float* mat_data = ifmaps;	
                        int src_off = j * stride * src_w + l * stride;
                        int ifmaps_idx = 0;		
						for(int sc = 0; sc < src_channels; ++sc){
							for (int sh = 0; sh < kernel_h; ++sh){
                                //cudaMemcpy(mat_data, A + src_off, kernel_w * sizeof(float), cudaMemcpyDeviceToDevice);
                                //mat_data += kernel_w;
                                for(int sw = 0; sw < kernel_w; ++sw){
                                    ifmaps[ifmaps_idx] = *(A + src_off);
                                    ifmaps_idx++;
                                    src_off++;
                                }
                                src_off += src_w - kernel_w;
							}
							src_off += src_w * src_h - src_w * kernel_h;
						}	
						//calculate one output element
						C[idx] = dev_dot(filter, ifmaps, k);			
					}
				}
				filter += k;
			}
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
		default: ;

	}
}
