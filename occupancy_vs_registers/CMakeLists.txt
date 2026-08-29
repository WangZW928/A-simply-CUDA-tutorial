cmake_minimum_required(VERSION 3.24)

project(occupancy_vs_registers LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(occupancy_bench src/occupancy_bench.cu)

target_compile_options(
    occupancy_bench
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xptxas=-v
            -gencode=arch=compute_89,code=sm_89
        >
)

target_link_libraries(occupancy_bench PRIVATE CUDA::cudart)
