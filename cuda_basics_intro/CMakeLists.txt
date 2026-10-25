cmake_minimum_required(VERSION 3.24)
# native 需要配置时能访问 GPU；无 GPU 的编译机请显式传入目标架构。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cuda_basics_intro LANGUAGES CXX CUDA)

add_executable(basics_bench src/basics_bench.cu)
set_target_properties(basics_bench PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(basics_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
