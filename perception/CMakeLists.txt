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

add_executable(loci_capture src/capture.cpp)
target_link_libraries(loci_capture PRIVATE realsense2::realsense2)
target_compile_options(loci_capture PRIVATE -Wall -Wextra -Wpedantic)
