#include <stdio.h>
#include <iostream>
#include <sstream>

#include <assert.h>

#include "cpu_solver.h"
#include "utils.h"

#include <gtest/gtest.h>
#include <iostream>

TEST(QuarticTest, HandlesKnownMins) {
    float *A, *B, *C, *D, *E;
	float *min_x;

	// malloc for cpu solver
	A = (float*)malloc(N * sizeof(float));
	B = (float*)malloc(N * sizeof(float));
	C = (float*)malloc(N * sizeof(float));
	D = (float*)malloc(N * sizeof(float));
	E = (float*)malloc(N * sizeof(float));
    
    /*A = new float();
    float A[]= {2, 14, 3};
    float B[]= {-4, -11, 87}; 
    float C[]= {-22, 51, 0}; 
    float D[]= {24, 79, 0}; 
    float E[]= {2, 1, 0};*/

	std::cout << "generating data..." << std::endl;

	generate_data(N, -100, 100, A);
	generate_data(N, -100, 100, B);
	generate_data(N, -100, 100, C);
	generate_data(N, -100, 100, D);
	generate_data(N, -100, 100, E);

	for (int i = 0; i < N; i++) {
		if (A[i] == 0) { A[i] = 1; } // done to avoid undefined behaviour in solver when A=0
	}
	cpu_quartic_minimum_helper(N, A, B, C, D, E, min);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}