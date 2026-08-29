cmake_minimum_required(VERSION 3.18)
project(kernel_type_playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(kernel_types src/kernel_types.cu)
set_target_properties(kernel_types PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
