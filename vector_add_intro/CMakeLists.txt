cmake_minimum_required(VERSION 3.24)
# 与基础章一致；无 GPU 的编译机可显式传入 -DCMAKE_CUDA_ARCHITECTURES=89。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(vector_add_intro LANGUAGES CXX CUDA)

add_executable(vector_add_demo src/vector_add_demo.cu)
set_target_properties(vector_add_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(vector_add_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
