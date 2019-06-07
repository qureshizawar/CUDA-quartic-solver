cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda-quartic-solver LANGUAGES CXX CUDA)

#include_directories("${CUDA_INCLUDE_DIRS}")

add_library(cuda-quartic-solver STATIC
  cpu_solver.cpp
  cpu_solver.h
  gpu_solver.cu
  gpu_solver.cuh
  utils.cpp
  utils.h
  )


# Request that cuda-quartic-solver be built with -std=c++11
# As this is a public compile feature anything that links to cuda-quartic-solver
# will also build with -std=c++11
target_compile_features(cuda-quartic-solver PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the cuda-quartic-solver
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( cuda-quartic-solver
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(quartic_solver_main main.cu)

  set_target_properties(quartic_solver_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(quartic_solver_main PRIVATE cuda-quartic-solver)