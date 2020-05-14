#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>  // for high_resolution_clock

#include "cpu_solver.h"
#include "gpu_solver.cuh"
#include "utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

void dry_run(int N){

        std::cout << "######################################################" << std::endl;
        std::cout << "A quick test to evaluate overall functionality and performance" << std::endl;
        std::cout << "######################################################" << std::endl;

        int testruns = 10;

        float *A, *B, *C, *D, *E;
        float *min;

        // host malloc

        checkCuda(cudaMallocHost((void**)&A, N * sizeof(float))); // host pinned
        checkCuda(cudaMallocHost((void**)&B, N * sizeof(float))); // host pinned
        checkCuda(cudaMallocHost((void**)&C, N * sizeof(float))); // host pinned
        checkCuda(cudaMallocHost((void**)&D, N * sizeof(float))); // host pinned
        checkCuda(cudaMallocHost((void**)&E, N * sizeof(float))); // host pinned

        checkCuda(cudaMallocHost((void**)&min, N * sizeof(float))); // host pinned

        memset(min, 0, N * sizeof(float));

        std::cout << "generating data..." << std::endl;

        generate_data(N, -100, 100, A);
        generate_data(N, -100, 100, B);
        generate_data(N, -100, 100, C);
        generate_data(N, -100, 100, D);
        generate_data(N, -100, 100, E);

        for (int i = 0; i < N; i++) {
                if (A[i] == 0) { A[i] = 1; } // done to avoid undefined behaviour in solver when A=0
        }

        std::cout << "done!" << std::endl;

        float dur = 0;
        float milliseconds = 0;
        float avg_cpu = 0;
        float avg = 0;

        std::cout << "####################### CPU ##########################" << std::endl;
        std::cout << "######################################################" << std::endl;
        std::cout << "######################################################" << std::endl;

        for (int k = 0; k < testruns; ++k) {
                auto pstart = std::chrono::high_resolution_clock::now();

                QuarticMinimumCPU(N, A, B, C, D, E, min);

                auto finish = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> elapsed = finish - pstart;
                dur = elapsed.count() * 1000;
                printf("Time (ms): %f\n", dur);
                avg_cpu += dur;
        }

        printf("min[0]: %f \n",min[0]);
        printf("avgTime (ms): %f\n", avg_cpu / testruns);

        memset(min, 0, N * sizeof(float));

        avg = 0;

        std::cout << "####################### GPU (no streams) #############" << std::endl;
        std::cout << "######################################################" << std::endl;
        std::cout << "######################################################" << std::endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int k = 0; k < testruns; ++k) {

                cudaEventRecord(start);

                QuarticMinimumGPU(N, A, B, C, D, E, min);

                cudaEventRecord(stop);

                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("Time (ms): %f\n", milliseconds);
                avg += milliseconds;
        }

        printf("min[0]: %f \n",min[0]);
        printf("avgTime (ms): %f\n", avg / testruns);

        avg = 0;

        std::cout << "####################### GPU (streams) ################" << std::endl;
        std::cout << "######################################################" << std::endl;
        std::cout << "######################################################" << std::endl;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int k = 0; k < testruns; ++k) {

                cudaEventRecord(start);

                QuarticMinimumGPUStreams(N, A, B, C, D, E, min);

                cudaEventRecord(stop);

                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("Time (ms): %f\n", milliseconds);
                avg += milliseconds;
        }

        printf("min[0]: %f \n",min[0]);
        printf("avgTime (ms): %f\n", avg / testruns);

        std::cout << "######################################################" << std::endl;
        std::cout << "######################################################" << std::endl;

        printf("Speedup Tcpu/Tgpu: %f \n", avg_cpu / avg);

        checkCuda(cudaFreeHost(A));
        checkCuda(cudaFreeHost(B));
        checkCuda(cudaFreeHost(C));
        checkCuda(cudaFreeHost(D));
        checkCuda(cudaFreeHost(E));
        checkCuda(cudaFreeHost(min));
}

template <typename T>
pybind11::array_t<T> QuarticMinimum(pybind11::array_t<T> A, pybind11::array_t<T> B,
                                    pybind11::array_t<T> C, pybind11::array_t<T> D, pybind11::array_t<T> E, bool use_gpu)
{
        pybind11::buffer_info bufA = A.request();
        pybind11::buffer_info bufB = B.request();
        pybind11::buffer_info bufC = C.request();
        pybind11::buffer_info bufD = D.request();
        pybind11::buffer_info bufE = E.request();

        if (bufA.ndim != 1 || bufB.ndim != 1 || bufC.ndim != 1 || bufD.ndim != 1 || bufE.ndim != 1)
                throw std::runtime_error("Number of dimensions must be one");

        if (bufA.size != bufB.size || bufA.size != bufC.size || bufA.size != bufD.size || bufA.size != bufE.size)
                throw std::runtime_error("Input shapes must match");

        int size = bufA.shape[0];
        //int size_bytes = size*sizeof(T);

        T* aptr = reinterpret_cast<T*>(bufA.ptr);
        T* bptr = reinterpret_cast<T*>(bufB.ptr);
        T* cptr = reinterpret_cast<T*>(bufC.ptr);
        T* dptr = reinterpret_cast<T*>(bufD.ptr);
        T* eptr = reinterpret_cast<T*>(bufE.ptr);

        /* No pointer is passed, so NumPy will allocate the buffer */
        auto result = pybind11::array_t<T>(bufA.size);
        pybind11::buffer_info bufRes = result.request();

        T* min_x = (T*) bufRes.ptr;

        if (use_gpu) {
                if(size<1000) {
                        //std::cout<<"using CPU"<<std::endl;
                        QuarticMinimumCPU(size, aptr, bptr, cptr, dptr, eptr, min_x);
                }
                else if (size<10000) {
                        //std::cout<<"using GPU"<<std::endl;
                        QuarticMinimumGPU(size, aptr, bptr, cptr, dptr, eptr, min_x);
                }
                else{
                        //std::cout<<"using GPU with streams"<<std::endl;
                        QuarticMinimumGPUStreams(size, aptr, bptr, cptr, dptr, eptr, min_x);
                }
        }
        else{
                //std::cout<<"using CPU"<<std::endl;
                QuarticMinimumCPU(size, aptr, bptr, cptr, dptr, eptr, min_x);
        }


        return result;
}

PYBIND11_MODULE(QuarticSolver, m) {
        m.doc() = R"pbdoc(
        Pybind11 QuarticSolver plugin
        -----------------------
        .. currentmodule:: QuarticSolver
        .. autosummary::
           :toctree: _generate
           dry_run
           QuarticMinimum
    )pbdoc";

        m.def("dry_run", &dry_run, R"pbdoc(
        A quick test to evaluate overall functionality and performance.
    )pbdoc");
        m.def("QuarticMinimum", &QuarticMinimum<float>, R"pbdoc(
        Takes the coefficients for a quartic function as a numpy array,
        calculates the corresponding minimums and returns results as a numpy array.
        Final arg is a boolean which sets whether to use the GPU or not.
    )pbdoc");
    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

int main(void)
{
        int N = (1 << 20);
        std::cout << "N = " << N << std::endl;
        dry_run(N);

}
