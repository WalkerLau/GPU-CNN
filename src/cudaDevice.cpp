#include <cuda_runtime.h>
#include <iostream>

void PrintGPU(){
    std::cout<<"..................................................... "<<std::endl;
    std::cout<<"GPU info: "<<std::endl;
    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);

    cudaDeviceProp dev_prop;
    for(int i = 0; i < dev_cnt; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        std::cout<<".......................GPU ["<<i+1<<"]....................... "<<std::endl;
        std::cout<<"multiprocessors(SMs)....  "<<dev_prop.multiProcessorCount<<std::endl;
        std::cout<<"totalGlobalMem..........  "<<dev_prop.totalGlobalMem/(1024*1024)<<" MB"<<std::endl;
        std::cout<<"sharedMemPerBlock.......  "<<dev_prop.sharedMemPerBlock<<std::endl;
        std::cout<<"regsPerBlock............  "<<dev_prop.regsPerBlock<<std::endl;
        std::cout<<"totalConstMem...........  "<<dev_prop.totalConstMem<<std::endl;
        std::cout<<"warpSize................  "<<dev_prop.warpSize<<std::endl;
        std::cout<<"maxThreadsPerSM.........  "<<dev_prop.maxThreadsPerMultiProcessor<<std::endl;
        std::cout<<"maxBlockDim(x)..........  "<<dev_prop.maxThreadsDim[0]<<std::endl;
        std::cout<<"maxBlockDim(y)..........  "<<dev_prop.maxThreadsDim[1]<<std::endl;
        std::cout<<"maxBlockDim(z)..........  "<<dev_prop.maxThreadsDim[2]<<std::endl;
        std::cout<<"maxGridDim(x)...........  "<<dev_prop.maxGridSize[0]<<std::endl;
        std::cout<<"maxGridDim(y)...........  "<<dev_prop.maxGridSize[1]<<std::endl;
        std::cout<<"maxGridDim(z)...........  "<<dev_prop.maxGridSize[2]<<std::endl;
    }
    std::cout<<"..................................................... "<<std::endl;
    
}


