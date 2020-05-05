cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(GPU_build)
    project(QuarticSolver LANGUAGES CXX CUDA)
    find_package(PythonInterp 3.6 REQUIRED)
    find_package(PythonLibs 3.6 REQUIRED)

    include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ./pybind11/include
        )

    add_library(QuarticSolver SHARED
      src/main.cu
      src/cpu_solver.cpp
      src/cpu_solver.h
      src/gpu_solver.cu
      src/gpu_solver.cuh
      src/utils.cpp
      src/utils.h
      )


    # Request that QuarticSolver be built with -std=c++11
    # As this is a public compile feature anything that links to QuarticSolver
    # will also build with -std=c++11
    target_compile_features(QuarticSolver PUBLIC cxx_std_11)

    target_compile_options(QuarticSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)

    # We need to explicitly state that we need all CUDA files in the QuarticSolver
    # library to be built with -dc as the member functions could be called by
    # other libraries and executables
    set_target_properties( QuarticSolver
                           PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                          )

    if( CPP_exe ) 
        # build a c++ executable
        add_executable(quartic_solver_main src/main.cu)
        set_target_properties(quartic_solver_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(quartic_solver_main PRIVATE QuarticSolver)
    endif()

    target_link_libraries(QuarticSolver
    ${PYTHON_LIBRARIES})

    set_target_properties(QuarticSolver PROPERTIES PREFIX "")

else()
    project(QuarticSolver CXX)
    find_package(PythonInterp 3.6 REQUIRED)
    find_package(PythonLibs 3.6 REQUIRED)

    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ./pybind11/include
        )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    add_library(QuarticSolver SHARED
      src/cpu_main.cpp
      src/cpu_solver.cpp
      src/cpu_solver.h
      src/utils.cpp
      src/utils.h)
      
      if( CPP_exe ) 
        # build a c++ executable
        add_executable(quartic_solver_main src/cpu_main.cpp)
        target_link_libraries(quartic_solver_main PRIVATE QuarticSolver)
      endif()

    target_link_libraries(QuarticSolver
      ${PYTHON_LIBRARIES})

    set_target_properties(QuarticSolver PROPERTIES PREFIX "")
endif()