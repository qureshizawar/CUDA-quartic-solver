#include <iostream>
#include <chrono>  // for high_resolution_clock

#include "cpu_solver.h"
#include "utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void dry_run(int N) {

    std::cout << "######################################################" << std::endl;
    std::cout << "A quick test to evaluate overall functionality and performance" << std::endl;
    std::cout << "######################################################" << std::endl;

    int testruns = 10;

    float *min_x;

    // malloc for cpu solver
    auto A = std::make_unique<float []>(N);
    auto B = std::make_unique<float []>(N);
    auto C = std::make_unique<float []>(N);
    auto D = std::make_unique<float []>(N);
    auto E = std::make_unique<float []>(N);

    min_x = (float *) malloc(N * sizeof(float));
    memset(min_x, 0, N * sizeof(float));

    std::cout << "generating data..." << std::endl;

    generate_data(N, -100, 100, A.get());
    generate_data(N, -100, 100, B.get());
    generate_data(N, -100, 100, C.get());
    generate_data(N, -100, 100, D.get());
    generate_data(N, -100, 100, E.get());

    for (int i = 0; i < N; i++) {
        if (A[i] == 0) { A[i] = 1; } // done to avoid undefined behaviour in solver when A=0
    }

    std::cout << "done!" << std::endl;

    float dur = 0;
    float avg_cpu = 0;

    std::cout << "####################### CPU ##########################" << std::endl;
    std::cout << "######################################################" << std::endl;
    std::cout << "######################################################" << std::endl;

    for (int k = 0; k < testruns; ++k) {
        auto pstart = std::chrono::high_resolution_clock::now();
        QuarticMinimumCPU(N, A.get(), B.get(), C.get(), D.get(), E.get(), min_x);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = finish - pstart;
        dur = elapsed.count() * 1000;
        std::cout << "Time (ms): " << dur << '\n';
        avg_cpu += dur;
    }

    std::cout << "x_min: " << min_x[0] << '\n';
    std::cout << "avgTime (ms): " << avg_cpu / testruns << '\n';

}

template<typename T>
pybind11::array_t<T> QuarticMinimum(pybind11::array_t<T> A, pybind11::array_t<T> B,
                                    pybind11::array_t<T> C, pybind11::array_t<T> D, pybind11::array_t<T> E,
                                    bool use_gpu) {
    pybind11::buffer_info ha = A.request();
    pybind11::buffer_info hb = B.request();
    pybind11::buffer_info hc = C.request();
    pybind11::buffer_info hd = D.request();
    pybind11::buffer_info he = E.request();

    if (ha.ndim != 1) {
        std::stringstream sstream;
        sstream << "ha.ndim != 1" << std::endl;
        sstream << "ha.ndim: " << ha.ndim << std::endl;
        throw std::runtime_error(sstream.str());
    }

    int size = ha.shape[0];

    T *aptr = static_cast<T *>(ha.ptr);
    T *bptr = static_cast<T *>(hb.ptr);
    T *cptr = static_cast<T *>(hc.ptr);
    T *dptr = static_cast<T *>(hd.ptr);
    T *eptr = static_cast<T *>(he.ptr);

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = pybind11::array_t<T>(ha.size);
    pybind11::buffer_info buf = result.request();

    T *min_x = (T *) buf.ptr;

    if (use_gpu) {
        std::stringstream esstream;
        esstream
                << "WARNING: use_gpu set to True, this package was built with CPU only option please build package with GPU to use GPU capability"
                << std::endl;
        std::cout << esstream.str() << std::endl;
    }

    QuarticMinimumCPU(size, aptr, bptr, cptr, dptr, eptr, min_x);

    return result;
}

PYBIND11_MODULE(QuarticSolver, m
) {
    m.

            doc() = R"pbdoc(
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

int main(void) {
    std::cout << "Starting dry run..." << std::endl;
    int N = (1 << 20);
    std::cout << "N = " << N << std::endl;
    dry_run(N);
}
