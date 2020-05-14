cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(GPU_build)
    project(QuarticSolver LANGUAGES CXX CUDA)
    include(FindOpenMP)
    #find_package(PythonInterp 3.6 REQUIRED)
    #find_package(PythonLibs 3.6 REQUIRED)

    if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

    #include_directories(
    #    ${PYTHON_INCLUDE_DIRS}
    #    ./pybind11/include
    #    )

    if( CPP_exe )
        # build a c++ executable

        add_library(QuarticSolver SHARED
          src/main.cu
          src/cpu_solver.cpp
          src/cpu_solver.h
          src/gpu_solver.cu
          src/gpu_solver.cuh
          src/utils.cpp
          src/utils.h
          )

        # Request that QuarticSolver be built with -std=c++14
        # As this is a public compile feature anything that links to QuarticSolver
        # will also build with -std=c++14
        target_compile_features(QuarticSolver PUBLIC cxx_std_14)

        target_compile_options(QuarticSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)

        # We need to explicitly state that we need all CUDA files in the QuarticSolver
        # library to be built with -dc as the member functions could be called by
        # other libraries and executables
        set_target_properties( QuarticSolver
                               PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                              )

        add_executable(quartic_solver_main src/main.cu)
        set_target_properties(quartic_solver_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(quartic_solver_main PRIVATE QuarticSolver)
    else()
      # build pip package
      add_subdirectory(pybind11)
      ##pybind11 sets -fvisibility=hidden in INTERFACE_COMPILE_OPTIONS on it's module target
      get_target_property(modifacecopts module INTERFACE_COMPILE_OPTIONS)
      list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
      set_target_properties(module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")

      add_library(QuarticSolver MODULE
      src/main_py.cu
      src/cpu_solver.cpp
      src/cpu_solver.h
      src/gpu_solver.cu
      src/gpu_solver.cuh
      src/utils.cpp
      src/utils.h
      )
      set_target_properties(QuarticSolver PROPERTIES
          POSITION_INDEPENDENT_CODE ON
          CUDA_VISIBILITY_PRESET "hidden"
          CXX_VISIBILITY_PRESET "hidden"
          INTERPROCEDURAL_OPTIMIZATION TRUE
          PREFIX "${PYTHON_MODULE_PREFIX}"
          SUFFIX "${PYTHON_MODULE_EXTENSION}"
      )

      target_link_libraries(QuarticSolver PRIVATE pybind11::module)

    endif()

else()
    project(QuarticSolver CXX)
    include(FindOpenMP)
    if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

    if( CPP_exe )
      # build a c++ executable

      add_library(QuarticSolver SHARED
        src/cpu_main.cpp
        src/cpu_solver.cpp
        src/cpu_solver.h
        src/utils.cpp
        src/utils.h)
      add_executable(quartic_solver_main src/cpu_main.cpp)
      target_link_libraries(quartic_solver_main PRIVATE QuarticSolver)
      set_target_properties(QuarticSolver PROPERTIES PREFIX "")

    else()
      # build pip package
      add_subdirectory(pybind11)
      pybind11_add_module(QuarticSolver
        src/cpu_main_py.cpp
        src/cpu_solver.cpp
        src/cpu_solver.h
        src/utils.cpp
        src/utils.h)


    endif()
endif()
