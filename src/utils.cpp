#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include <stdio.h>
#include <iostream>
#include <stack>
#include <ctime>

#include <chrono>  // for high_resolution_clock

#include <assert.h>

using namespace std;

clock_t start;
double duration;

void tic() {
	start = clock();
}

void toc() {
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	printf("Time(s):  %f \n", duration);
}

void generate_data(size_t size, int min, int max, float *data) {

	random_device rd; //seed generator
	mt19937_64 generator{ rd() }; //generator initialized with seed from rd
	//std::normal_distribution<> d{mean,sd};
	uniform_int_distribution<> dist{ min, max };
	for (int i = 0; i<size; ++i) {
		data[i] = dist(generator);
	}

}