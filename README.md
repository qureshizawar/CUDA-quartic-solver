# CUDA-quartic-solver

A general cubic equation solver and quartic equation minimisation solver written for CPU and Nvidia GPUs, for more details and results, see: https://arxiv.org/abs/1903.10041

## Running the solver

### Requirements

1) CUDA 9.0+ capable GPU required to run the GPU solvers
2) CMake 3.8+ (if using CMake) 

### MS Visual Studio

MS visual studio solution file ```cuda-quartic-solver.sln``` provided to run the solver and example code

### CMake

```CMakeLists.txt``` file provided inside the ```quartic_solver``` folder, CMake can be used to build the project for example:

```
cd quartic_solver
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ../
make
```

Then ```quartic_solver_main``` can be executed to run the example code comparing the CPU and GPU quartic solver execution times 

## TODO

1) Add multi-gpu support
