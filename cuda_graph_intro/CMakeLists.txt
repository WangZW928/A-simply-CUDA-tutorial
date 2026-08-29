cmake_minimum_required(VERSION 3.18)
project(cuda_graph_intro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(graph_bench src/graph_bench.cu)
set_target_properties(graph_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
