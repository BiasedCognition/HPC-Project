cmake_minimum_required(VERSION 3.18)
project(HPC_CUDA_PartA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(rotate_part_a
  src/main.cu
  src/geom.cpp
  src/rotate.cu
)

target_include_directories(rotate_part_a PRIVATE include)
target_link_libraries(rotate_part_a PRIVATE CUDA::cudart)

if(MSVC)
  target_compile_options(rotate_part_a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
else()
  target_compile_options(rotate_part_a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
endif()

set_target_properties(rotate_part_a PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "native"
)
