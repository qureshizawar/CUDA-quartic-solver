# CUDA-quartic-solver

![GitHub](https://img.shields.io/github/license/qureshizawar/quartic-solver)

A general cubic equation solver and quartic equation minimisation solver written for CPU and Nvidia GPUs, for more details and results, see: https://arxiv.org/abs/1903.10041. The library is available for C++/CUDA as well as Python using Pybind11.

## Running the solver

### Requirements

1) CUDA 9.0+ capable GPU required to run the GPU solvers
2) CMake 3.8+ 
3) Python 3.6, numpy (if using Python)

### CMake

CMake can be used to build the project, for example:
```
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ../ -D CPP_exe=true -D GPU_build=false
make
```

When the `CPP_exe` flag is set to `true`, CMake will build a c++ executable, then ```quartic_solver_main``` can be executed to run the example code.
When the `GPU_build` flag is set to `true`, CMake will build the CUDA capable version of the library

### Python

By default, CMake will also build the Python version of the library which can be directly imported from the `build` folder

The Python package can also be installed by building its `.whl` file, for example:
```
python setup.py bdist_wheel --GPU_build True
```
When the `GPU_build` flag is set to `True`, the CUDA capable version of the library will be built
The built `.whl` can then be used to install the package, for example:
```
pip install ./dist/QuarticSolver-0.1.1-cp36-cp36m-linux_x86_64.whl
```

### Usage
Please see `src/main.cu`, `src/cpu_main.cpp`, and the examples in the `/tests` folder for detailed usage examples
##### C++
Given quartic functions of the form `Ax^4 + Bx^3 + Cx^2 + Dx + E` the minimiser for the quartic functions can calculated on the CPU by calling the following function from `cpu_solver.cpp`:
```
QuarticMinimumCPU(N, A, B, C, D, E, min);
```
Where:
- `N` is the number of functions
- `A` is an array containing the coefficient A for each of the quartic functions
- `B` is an array containing the coefficient B for each of the quartic functions
- `C` is an array containing the coefficient C for each of the quartic functions
- `D` is an array containing the coefficient D for each of the quartic functions
- `E` is an array containing the coefficient E for each of the quartic functions
- `min` is an array which will contain the corresponding minimums for each of the quartic functions

The calculations can be carried out on a GPU by using `QuarticMinimumGPU` or `QuarticMinimumGPUStreams`. Please note signifcant performance improvement is observed when `N>10000` for `QuarticMinimumGPU` vs `QuarticMinimumGPUStreams`

##### Python
The Python version of the library can be used as follows:
```
import numpy as np
import QuarticSolver

minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
```
Where `A`,`B`,`C`,`D`,`E` are numpy arrays containing the quartic function coefficients and the final arg is a boolean flag which if set to `True` will use the GPU if possible. `minimum` is the returned numpy array containing the corresponding minimisers.
## TODO

1) Add multi-gpu support
