cmake_minimum_required(VERSION 3.22)
project(loci_perception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# librealsense v2.59.1, built from source and installed to /usr/local
find_package(realsense2 REQUIRED)
# CUDA runtime and TensorRT from JetPack (CUDA in /usr/local/cuda, TensorRT in the system paths)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES aarch64-linux-gnu REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)

set(LOCI_WARNINGS -Wall -Wextra -Wpedantic)

add_executable(loci_capture src/capture.cpp)
target_link_libraries(loci_capture PRIVATE realsense2::realsense2)
target_compile_options(loci_capture PRIVATE ${LOCI_WARNINGS})

add_library(loci_detector src/detector.cpp)
target_include_directories(loci_detector PUBLIC src)
target_include_directories(loci_detector SYSTEM PUBLIC ${TENSORRT_INCLUDE_DIR})
target_link_libraries(loci_detector PUBLIC CUDA::cudart ${NVINFER_LIBRARY})
target_compile_options(loci_detector PRIVATE ${LOCI_WARNINGS})

add_executable(loci_detect src/detect_main.cpp)
target_link_libraries(loci_detect PRIVATE loci_detector realsense2::realsense2)
target_compile_options(loci_detect PRIVATE ${LOCI_WARNINGS})
