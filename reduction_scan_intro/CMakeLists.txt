cmake_minimum_required(VERSION 3.18)
project(reduction_scan_intro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(reduce_scan_bench src/reduce_scan_bench.cu)
set_target_properties(reduce_scan_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
