# Accelerate CNN computation with GPU
Please indicate the source below when this repo is shared somewhere else:

Source code:  <https://github.com/WalkerLau/GPU-CNN>
 
The purpose of this repo is to accelerate VIPLFaceNet, a face recognition algorithm with 7 convolution layers, with GPU and CUDA programming.

For more infomation about VIPLFaceNet:

<https://github.com/seetaface/SeetaFaceEngine>

# Development Environment

* **HARDWARE**
    
    * **Machine**: Amazon AWS EC2, g3s.xlarge (Ohio, us-east-2)
  
    * **GPU**: Tesla M60
  
    * **Compute Capability**: 5.2
  
* **SOFTWARE**
    * **AMI ID**: Deep Learning Base AMI (Ubuntu) Version 19.1 (ami-04fde1417a0168c95)
  
    * Ubuntu 16.04 (contained in AMI)
  
    * OpenCV 2.4.9.1 (contained in AMI)
  
    * CUDA 10.0.130 (contained in AMI)
  
# Installation

1. Download this repo
   
   `git clone https://github.com/WalkerLau/GPU-CNN.git`

   **Note:** The model file `GPU-CNN/model/seeta_fr_v1.0.bin` is stored in **Git Large File Storage (LFS)**. You should [Download and install Git Large File Storage (LFS)](https://git-lfs.github.com/) before further operations on Linux machine, or download the `seeta_fr_v1.0.bin` independently on this repo's webpage. **Simply "git clone" would not download the complete `seeta_fr_v1.0.bin` and will cause error**. The complete `seeta_fr_v1.0.bin` should be of about 110 MB.

2. Build
   
   ```
   cd GPU-CNN

   mkdir build

   cd build

   cmake ../src

   make
   ```

   **Note:** Try switching to root if build failed.

3. Run
   
   `./GPU-CNN`

# Others

* About source files:
  
  * **test_face_recognizer.cpp** contains top level function
  
  * **conv_net.cpp** calls convolution layers
  
  * **inner_product_net.cpp** calls fully-connected layers
  
  * **math_functions.cu** defines CUDA kernel functions
   

# References

* [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

* [Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
  
* [CUDA API References](https://docs.nvidia.com/cuda/index.html#cuda-api-references)
