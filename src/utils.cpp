#include "utils.h"


clock_t start;
double duration;

void tic() {
    start = clock();
}

void toc() {
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "Time(s): " << duration << '\n';
}

void generate_data(size_t size, int min, int max, float *data) {

    std::random_device rd; //seed generator
    std::mt19937_64 generator{rd()}; //generator initialized with seed from rd
    std::uniform_int_distribution<> dist{min, max};
    for (int i = 0; i < size; ++i) {
        data[i] = dist(generator);
    }

}