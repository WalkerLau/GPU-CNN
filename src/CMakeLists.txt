cmake_minimum_required (VERSION 3.0)
project (GPU-CNN)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

#message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
#message("Found OpenCV or not: " ${OpenCV_FOUND})
#message("opencv version: " ${OpenCV_VERSION})
#message("opencv dir: " ${OpenCV_DIR})
#message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
#message("OpenCV_LIBRARY_DIRS: " ${OpenCV_LIBRARY_DIRS})
#message("OpenCV_LIBRARIES : " ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_20,code=sm_20
  -gencode arch=compute_20,code=sm_21
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  -rdc=true
  )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

file(GLOB SRC 
    "${PROJECT_SOURCE_DIR}/*.cpp"
    "${PROJECT_SOURCE_DIR}/*.cu"
)

cuda_add_executable(GPU-CNN ${SRC})

target_include_directories(GPU-CNN PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}   
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(GPU-CNN PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)


