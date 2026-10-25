cmake_minimum_required(VERSION 3.24)
# 与基础章一致；无 GPU 的编译机请显式传入目标架构。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matrix_mul_intro LANGUAGES CXX CUDA)

add_executable(matrix_mul_demo src/matrix_mul_demo.cu)
set_target_properties(matrix_mul_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(matrix_mul_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
