#include <iostream>
#include <chrono>  // for high_resolution_clock

#include "cpu_solver.h"
#include "utils.h"

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

int main(void) {
        std::cout << "Starting dry run..." << std::endl;
        int N = (1 << 20);
        std::cout << "N = " << N << std::endl;
        dry_run(N);
}
