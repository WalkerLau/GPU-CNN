cmake_minimum_required (VERSION 3.0)
project (GPU-CNN)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

find_package( OpenCV REQUIRED )

#message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
#message("Found OpenCV or not: " ${OpenCV_FOUND})
#message("opencv version: " ${OpenCV_VERSION})
#message("opencv dir: " ${OpenCV_DIR})
#message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
#message("OpenCV_LIBRARY_DIRS: " ${OpenCV_LIBRARY_DIRS})
#message("OpenCV_LIBRARIES : " ${OpenCV_LIBS})

file(GLOB SRC "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable(GPU-CNN ${SRC})

target_include_directories(GPU-CNN PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}    
)

target_link_libraries(GPU-CNN PUBLIC
    ${OpenCV_LIBS}
)


