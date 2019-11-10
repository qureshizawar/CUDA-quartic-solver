#include <stdio.h>
#include <iostream>
#include <sstream>
#include <assert.h>

#include <chrono>  // for high_resolution_clock

#include "cpu_solver.h"
#include "utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


using namespace std;
//__device__ __constant__ float d_pis[2];

void _test(int N){
    
    std::cout << "######################################################" << std::endl;
    std::cout << "A quick test to evaluate overall functionality and performance" << std::endl;
	std::cout << "######################################################" << std::endl;

	int testruns = 10;

	int erridx = 0;

	float *A, *B, *C, *D, *E;
	float *min_x;

	// malloc for cpu solver
	A = (float*)malloc(N * sizeof(float));
	B = (float*)malloc(N * sizeof(float));
	C = (float*)malloc(N * sizeof(float));
	D = (float*)malloc(N * sizeof(float));
	E = (float*)malloc(N * sizeof(float));
    
    min_x = (float*)malloc(N * sizeof(float));
	memset(min_x, 0, N * sizeof(float));

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

	bool debug = true;

	std::cout << "####################### CPU ##########################" << std::endl;
	std::cout << "######################################################" << std::endl;
	std::cout << "######################################################" << std::endl;

	for (int k = 0; k < testruns; ++k) {
		auto pstart = std::chrono::high_resolution_clock::now();
		QuarticMinimumCPU(N, A, B, C, D, E, min_x);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = finish - pstart;
		dur = elapsed.count() * 1000;
		printf("Time (ms): %f\n", dur);
		avg_cpu += dur;
	}

	printf("x_min: %f \n", min_x[0]);
	printf("avgTime (ms): %f\n", avg_cpu / testruns);
}

template <typename T>
pybind11::array_t<T> QuarticMinimum(pybind11::array_t<T> A, pybind11::array_t<T> B,
	pybind11::array_t<T> C, pybind11::array_t<T> D, pybind11::array_t<T> E, bool use_gpu)
{
	pybind11::buffer_info ha = A.request();
	pybind11::buffer_info hb = B.request();
	pybind11::buffer_info hc = C.request();
	pybind11::buffer_info hd = D.request();
	pybind11::buffer_info he = E.request();

	if (ha.ndim != 1) {
	std::stringstream strstr;
	strstr << "ha.ndim != 1" << std::endl;
	strstr << "ha.ndim: " << ha.ndim << std::endl;
	throw std::runtime_error(strstr.str());
	}

	int size = ha.shape[0];
	//int size_bytes = size*sizeof(T);

	T* aptr = reinterpret_cast<T*>(ha.ptr);
	T* bptr = reinterpret_cast<T*>(hb.ptr);
	T* cptr = reinterpret_cast<T*>(hc.ptr);
	T* dptr = reinterpret_cast<T*>(hd.ptr);
	T* eptr = reinterpret_cast<T*>(he.ptr);

	/* No pointer is passed, so NumPy will allocate the buffer */
	auto result = pybind11::array_t<T>(ha.size);
	pybind11::buffer_info buf = result.request();

	T* min_x = (T*) buf.ptr;
    
    if(use_gpu){
        std::cout<<"WARNING: use_gpu set to True, this package was built with CPU only option please build package with GPU to use GPU capability"
            <<std::endl;
    }

	QuarticMinimumCPU(size, aptr, bptr, cptr, dptr, eptr, min_x);

	/*for(int i=0; i<size; i++){
		std::cout<<"min: "<<min_x[i]<<std::endl;
	}*/

	return result;
}

PYBIND11_MODULE(QuarticSolver, m) {
    m.doc() = R"pbdoc(
        Pybind11 QuarticSolver plugin
        -----------------------
        .. currentmodule:: QuarticSolver
        .. autosummary::
           :toctree: _generate
           _test
           QuarticMinimum
    )pbdoc";

    m.def("_test", &_test, R"pbdoc(
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
	
	_test(N);

}