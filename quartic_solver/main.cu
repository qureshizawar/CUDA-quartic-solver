#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <stdio.h>
#include <iostream>
#include <stack>
#include <ctime>
#include <chrono>  // for high_resolution_clock
#include <assert.h>

#include "cpu_solver.h"
#include "gpu_solver.cuh"
#include "utils.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

//__device__ __constant__ float d_pis[2];

int main(void)
{
	int N = (1 << 20);
	std::cout << "N = " << N << std::endl;

	int testruns = 10;
	int debg = 0;

	int erridx = 0;
	// init CUDA streams
	const int nStreams = 4;
	const int streamSize = N / nStreams;
	const int streamBytes = streamSize * sizeof(float);

	// kernel dims
	int block_dim = 64;

	float *A, *B, *C, *D, *E, *bi, *ci, *di, *h_theta, *h_sqrtQ, *x1, *x2, *x3, *min, *Q, *R, *Qint, *Rint, *del, *h_temp;
	float *x1_cpu, *x2_cpu, *x3_cpu, *min_cpu;
	float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ, *d_Q, *d_R, *d_Qint, *d_Rint, *d_del,
		*d_x1, *d_x2, *d_x3, *d_min, *d_temp;
	//float *x1_img, *x2_img, *x3_img, *d_x1_img, *d_x2_img, *d_x3_img;

	// host malloc

	checkCuda(cudaMallocHost((void**)&A, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&B, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&C, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&D, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&E, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&x1, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&x2, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&x3, N * sizeof(float))); // host pinned

	checkCuda(cudaMallocHost((void**)&Q, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&R, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&Qint, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&Rint, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&del, N * sizeof(float))); // host pinned

	checkCuda(cudaMallocHost((void**)&h_temp, N * sizeof(float))); // host pinned

	checkCuda(cudaMallocHost((void**)&min, N * sizeof(float))); // host pinned

	/*checkCuda(cudaMallocHost((void**)&x1_img, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&x2_img, N * sizeof(float))); // host pinned
	checkCuda(cudaMallocHost((void**)&x3_img, N * sizeof(float))); // host pinned*/

	// device malloc

	checkCuda(cudaMalloc(&d_A, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_B, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_C, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_D, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_E, N * sizeof(float)));

	checkCuda(cudaMalloc(&d_bi, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_ci, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_di, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_theta, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_sqrtQ, N * sizeof(float)));

	checkCuda(cudaMalloc(&d_Q, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_R, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_Qint, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_Rint, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_del, N * sizeof(float)));

	checkCuda(cudaMalloc(&d_x1, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_x2, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_x3, N * sizeof(float)));

	checkCuda(cudaMalloc(&d_min, N * sizeof(float)));

	checkCuda(cudaMalloc(&d_temp, N * sizeof(float)));


	// malloc for cpu solver

	bi = (float*)malloc(N * sizeof(float));
	ci = (float*)malloc(N * sizeof(float));
	di = (float*)malloc(N * sizeof(float));

	h_theta = (float*)malloc(N * sizeof(float));
	h_sqrtQ = (float*)malloc(N * sizeof(float));

	x1_cpu = (float*)malloc(N * sizeof(float));
	x2_cpu = (float*)malloc(N * sizeof(float));
	x3_cpu = (float*)malloc(N * sizeof(float));

	min_cpu = (float*)malloc(N * sizeof(float));

	memset(x1_cpu, 0, N * sizeof(float));
	memset(x2_cpu, 0, N * sizeof(float));
	memset(x3_cpu, 0, N * sizeof(float));

	memset(min_cpu, 0, N * sizeof(float));

	//debug
	/*float *de1, *de2, *de3, *de4;

	de1 = (float*)malloc(N * sizeof(float));
	de2 = (float*)malloc(N * sizeof(float));
	de3 = (float*)malloc(N * sizeof(float));
	de4 = (float*)malloc(N * sizeof(float));*/
	
	// malloc for error checking between gpu and cpu solvers
	// some error is inevitiatble esp. for fuctions such as sqrt/cbrt and trig functions (cosf etc.)
	// see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix, Mathematical Functions, for more

	float *errx;

	errx = (float*)malloc(4 * sizeof(float));

	memset(errx, 0, 4 * sizeof(float));

	/*checkCuda(cudaMalloc(&d_x1_img, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_x2_img, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_x3_img, N * sizeof(float)));*/

	std::cout << "generating data..." << std::endl;

	generate_data(N, -100, 100, A);
	generate_data(N, -100, 100, B);
	generate_data(N, -100, 100, C);
	generate_data(N, -100, 100, D);
	generate_data(N, -100, 100, E);

	for (int i = 0; i < N; i++) {
		if (A[i] == 0) { A[i] = 1; } // done to avoid undefined behaviour in solver when A=0
	}

	// debug cases
	//for (int i = 0; i < N; i++) {
		/*A[i] = 2.0f;
		B[i] = -4.0f;
		C[i] = -22.0f;
		D[i] = 24.0f;*/

		/*A[i] = 3;
		B[i] = 87;
		C[i] = 0;
		D[i] = 0;*/

		/*A[i] = -14;
		B[i] = -11;
		C[i] = 51;
		D[i] = 79;*/
	//}

	std::cout << "done!" << std::endl;

	float dur = 0;
	float milliseconds = 0;
	float avg_cpu = 0;
	float avg = 0;

	std::cout << "####################### CPU ######################" << std::endl;
	std::cout << "######################################################" << std::endl;
	std::cout << "######################################################" << std::endl;

	for (int k = 0; k < testruns; ++k) {
		auto pstart = std::chrono::high_resolution_clock::now();

		quarticSolver_cpu(N, A, B, C, D, bi, ci, di, Q, R, Qint, Rint, del, h_theta, h_sqrtQ, x1_cpu, x2_cpu, x3_cpu, h_temp, min_cpu);

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = finish - pstart;
		dur = elapsed.count() * 1000;
		printf("Time (ms): %f\n", dur);
		avg_cpu += dur;
	}

	/*printf("x1: %f x2: %f x3: %f \n", x1_cpu[0], x2_cpu[0], x3_cpu[0]);
	printf("x1: %f x2: %f x3: %f \n", x1_cpu[N - 1], x2_cpu[N - 1], x3_cpu[N - 1]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[0], x2_img[0], x3_img[0]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[N - 1], x2_img[N - 1], x3_img[N - 1]);*/

	printf("x1[0]: %f, x2[0]: %f , x3[0]: %f , min[0]: %f \n", x1_cpu[0], x2_cpu[0], x3_cpu[0], min_cpu[0]);

	if (debg) {

		printf("Q[0]: %f, R[0]: %f , del[0]: %f \n", Q[0], R[0], del[0]);

		printf("f(x_1):  %f \n", A[0] * (powf(x1_cpu[0], 4)) / 4 + B[0] * (powf(x1_cpu[0], 3)) / 3 + C[0] * (powf(x1_cpu[0], 2)) / 2 + D[0] * (x1_cpu[0]) + E[0]);
		printf("f(x_3):  %f \n", A[0] * (powf(x3_cpu[0], 4)) / 4 + B[0] * (powf(x3_cpu[0], 3)) / 3 + C[0] * (powf(x3_cpu[0], 2)) / 2 + D[0] * (x3_cpu[0]) + E[0]);

	}

	printf("avgTime (ms): %f\n", avg_cpu / testruns);
	//printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
	
	memset(x1, 0, N * sizeof(float));
	memset(x2, 0, N * sizeof(float));
	memset(x3, 0, N * sizeof(float));

	memset(min, 0, N * sizeof(float));

	avg = 0;

	/*memset(x1_img, 0, N * sizeof(float));
	memset(x2_img, 0, N * sizeof(float));
	memset(x3_img, 0, N * sizeof(float));*/


	cudaStream_t stream[nStreams + 1];

	for (int i = 0; i < nStreams; ++i) {
		checkCuda(cudaStreamCreate(&stream[i]));
	}
	checkCuda(cudaStreamCreate(&stream[nStreams]));

	std::cout << "####################### GPU (no streams) ######################" << std::endl;
	std::cout << "######################################################" << std::endl;
	std::cout << "######################################################" << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int k = 0; k < testruns; ++k) {

		cudaEventRecord(start);

		cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_D, D, N * sizeof(float), cudaMemcpyHostToDevice);

		//cudaEventRecord(start);

		QRdel << <(N + block_dim - 1) / block_dim, block_dim >> > (N, d_A, d_B, d_C, d_D, d_bi, d_ci, d_di, d_Q, d_R, d_Qint, d_Rint, d_del);

		QuarticSolver << <(N + block_dim - 1) / block_dim, block_dim >> > (N, d_A, d_B, d_C, d_D, d_bi, d_Q, d_R, d_del, d_theta, d_sqrtQ,
			d_x1, d_x2, d_x3, d_temp, d_min);

		//cudaEventRecord(stop);

		cudaMemcpy(x1, d_x1, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(x2, d_x2, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(x3, d_x3, N * sizeof(float), cudaMemcpyDeviceToHost);

		/*cudaMemcpy(de1, d_Q, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(de2, d_R, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(de3, d_del, N * sizeof(float), cudaMemcpyDeviceToHost);*/

		/*cudaMemcpy(x1_img, d_x1_img, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(x2_img, d_x2_img, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(x3_img, d_x3_img, N * sizeof(float), cudaMemcpyDeviceToHost);*/

		cudaMemcpy(min, d_min, N * sizeof(float), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time (ms): %f\n", milliseconds);
		avg += milliseconds;
	}

	/*printf("x1: %f x2: %f x3: %f \n", x1[0], x2[0], x3[0]);
	printf("x1: %f x2: %f x3: %f \n", x1[N - 1], x2[N - 1], x3[N - 1]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[0], x2_img[0], x3_img[0]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[N - 1], x2_img[N - 1], x3_img[N - 1]);*/

	printf("x1[0]: %f, x2[0]: %f , x3[0]: %f , min[0]: %f \n", x1[0], x2[0], x3[0], min[0]);

	if (debg) {

		printf("f(x_1):  %f \n", A[0] * (powf(x1[0], 4)) / 4 + B[0] * (powf(x1[0], 3)) / 3 + C[0] * (powf(x1[0], 2)) / 2 + D[0] * (x1[0]) + E[0]);
		printf("f(x_3):  %f \n", A[0] * (powf(x3[0], 4)) / 4 + B[0] * (powf(x3[0], 3)) / 3 + C[0] * (powf(x3[0], 2)) / 2 + D[0] * (x3[0]) + E[0]);

	}

	printf("avgTime (ms): %f\n", avg / testruns);
	//printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);

	if (debg) {

		memset(errx, 0, 4 * sizeof(float));

		for (int i = 0; i < N; ++i) {

			if (std::fabs(errx[3]) < std::fabs(min[i]) - std::fabs(min_cpu[i])) { errx[3] = std::fabs(min[i]) - std::fabs(min_cpu[i]); erridx = i; }

		}

		std::cout << erridx << std::endl;
		errx[0] = std::fabs(x1[erridx]) - std::fabs(x1_cpu[erridx]);
		errx[1] = std::fabs(x2[erridx]) - std::fabs(x2_cpu[erridx]);
		errx[2] = std::fabs(x3[erridx]) - std::fabs(x3_cpu[erridx]);
		printf("errx1: %f errx2: %f errx3: %f errmin: %f \n", errx[0], errx[1], errx[2], errx[3]);

		std::cout << "######################################################" << std::endl;
		std::cout << "######################################################" << std::endl;

		printf("Q[erridx]: %f, R[erridx]: %f , del[erridx]: %f \n", Q[erridx], R[erridx], del[erridx]);

		//printf("Q[0]: %f, R[0]: %f , del[0]: %f \n", de1[erridx], de2[erridx], de3[erridx]);

		std::cout << "######################################################" << std::endl;
		std::cout << "######################################################" << std::endl;

		printf("x1_cpu[erridx]: %f, x2_cpu[erridx]: %f , x3_cpu[erridx]: %f , min_cpu[erridx]: %f \n", x1_cpu[erridx], x2_cpu[erridx], x3_cpu[erridx], min_cpu[erridx]);

		std::cout << "checking f(x), (only useful when 3 real roots)" << std::endl;

		printf("f(x_1):  %f \n", A[erridx] * (powf(x1_cpu[erridx], 4)) / 4 + B[erridx] * (powf(x1_cpu[erridx], 3)) / 3 + C[erridx] * (powf(x1_cpu[erridx], 2)) / 2 + D[erridx] * (x1_cpu[erridx]) + E[erridx]);
		printf("f(x_3):  %f \n", A[erridx] * (powf(x3_cpu[erridx], 4)) / 4 + B[erridx] * (powf(x3_cpu[erridx], 3)) / 3 + C[erridx] * (powf(x3_cpu[erridx], 2)) / 2 + D[erridx] * (x3_cpu[erridx]) + E[erridx]);

		std::cout << "######################################################" << std::endl;

		printf("x1[erridx]: %f, x2[erridx]: %f , x3[erridx]: %f , min[erridx]: %f \n", x1[erridx], x2[erridx], x3[erridx], min[erridx]);

		std::cout << "checking f(x), (only useful when 3 real roots)" << std::endl;

		printf("f(x_1):  %f \n", A[erridx] * (powf(x1[erridx], 4)) / 4 + B[erridx] * (powf(x1[erridx], 3)) / 3 + C[erridx] * (powf(x1[erridx], 2)) / 2 + D[erridx] * (x1[erridx]) + E[erridx]);
		printf("f(x_3):  %f \n", A[erridx] * (powf(x3[erridx], 4)) / 4 + B[erridx] * (powf(x3[erridx], 3)) / 3 + C[erridx] * (powf(x3[erridx], 2)) / 2 + D[erridx] * (x3[erridx]) + E[erridx]);


		std::cout << "######################################################" << std::endl;

		printf("A[erridx]: %f, B[erridx]: %f , C[erridx]: %f , D[erridx]: %f \n", A[erridx], B[erridx], C[erridx], D[erridx]);

		std::cout << "######################################################" << std::endl;
		std::cout << "######################################################" << std::endl;

	}

	memset(x1, 0, N * sizeof(float));
	memset(x2, 0, N * sizeof(float));
	memset(x3, 0, N * sizeof(float));

	avg = 0;

	/*memset(x1_img, 0, N * sizeof(float));
	memset(x2_img, 0, N * sizeof(float));
	memset(x3_img, 0, N * sizeof(float));*/

	memset(errx, 0, 4 * sizeof(float));

	std::cout << "####################### GPU (streams) ######################" << std::endl;
	std::cout << "######################################################" << std::endl;
	std::cout << "######################################################" << std::endl;

	int offset = 0;
	const int resstreamSize = N % nStreams;
	const int resstreamBytes = resstreamSize * sizeof(float);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int k = 0; k < testruns; ++k) {

		cudaEventRecord(start);

		for (int i = 0; i < nStreams; ++i) {
			offset = i * streamSize;
			cudaMemcpyAsync(&d_A[offset], &A[offset],
				streamBytes, cudaMemcpyHostToDevice, stream[i]);
			cudaMemcpyAsync(&d_B[offset], &B[offset],
				streamBytes, cudaMemcpyHostToDevice, stream[i]);
			cudaMemcpyAsync(&d_C[offset], &C[offset],
				streamBytes, cudaMemcpyHostToDevice, stream[i]);
			cudaMemcpyAsync(&d_D[offset], &D[offset],
				streamBytes, cudaMemcpyHostToDevice, stream[i]);

			QRdel << <(streamSize + block_dim - 1) / block_dim, block_dim, 0, stream[i] >> > (streamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
				&d_bi[offset], &d_ci[offset], &d_di[offset], &d_Q[offset], &d_R[offset], &d_Qint[offset], &d_Rint[offset], &d_del[offset]);
			
			QuarticSolver << <(streamSize + block_dim - 1) / block_dim, block_dim, 0, stream[i] >> > (streamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
				&d_bi[offset], &d_Q[offset], &d_R[offset], &d_del[offset], &d_theta[offset], &d_sqrtQ[offset], &d_x1[offset], &d_x2[offset], &d_x3[offset],
				&d_temp[offset], &d_min[offset]);

			cudaMemcpyAsync(&x1[offset], &d_x1[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x2[offset], &d_x2[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x3[offset], &d_x3[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);

			/*cudaMemcpyAsync(&x1_img[offset], &d_x1_img[offset],
			streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x2_img[offset], &d_x2_img[offset],
			streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x3_img[offset], &d_x3_img[offset],
			streamBytes, cudaMemcpyDeviceToHost, stream[i]);*/

			cudaMemcpyAsync(&min[offset], &d_min[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);
		}
		if (N%nStreams != 0) { // Catch last bit of data from potential unequal division between streams
			offset = nStreams * streamSize;

			cudaMemcpyAsync(&d_A[offset], &A[offset],
			resstreamBytes, cudaMemcpyHostToDevice, stream[nStreams]);
			cudaMemcpyAsync(&d_B[offset], &B[offset],
			resstreamBytes, cudaMemcpyHostToDevice, stream[nStreams]);
			cudaMemcpyAsync(&d_C[offset], &C[offset],
			resstreamBytes, cudaMemcpyHostToDevice, stream[nStreams]);
			cudaMemcpyAsync(&d_D[offset], &D[offset],
			resstreamBytes, cudaMemcpyHostToDevice, stream[nStreams]);

			QRdel << <(streamSize + block_dim - 1) / block_dim, block_dim, 0, stream[nStreams] >> > (streamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
				&d_bi[offset], &d_ci[offset], &d_di[offset], &d_Q[offset], &d_R[offset], &d_Qint[offset], &d_Rint[offset], &d_del[offset]);

			QuarticSolver << <(streamSize + block_dim - 1) / block_dim, block_dim, 0, stream[nStreams] >> > (streamSize, &d_A[offset], &d_B[offset], &d_C[offset], &d_D[offset],
				&d_bi[offset], &d_Q[offset], &d_R[offset], &d_del[offset], &d_theta[offset], &d_sqrtQ[offset], &d_x1[offset], &d_x2[offset], &d_x3[offset],
				&d_temp[offset], &d_min[offset]);

			cudaMemcpyAsync(&x1[offset], &d_x1[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
			cudaMemcpyAsync(&x2[offset], &d_x2[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
			cudaMemcpyAsync(&x3[offset], &d_x3[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);

			/*cudaMemcpyAsync(&x1_img[offset], &d_x1_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x2_img[offset], &d_x2_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x3_img[offset], &d_x3_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);*/

			cudaMemcpyAsync(&min[offset], &d_min[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
		}
		
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time (ms): %f\n", milliseconds);
		avg += milliseconds;
	}

	/*printf("x1: %f x2: %f x3: %f \n", x1[0], x2[0], x3[0]);
	printf("x1: %f x2: %f x3: %f \n", x1[N - 1], x2[N - 1], x3[N - 1]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[0], x2_img[0], x3_img[0]);
	printf("x1_img: %f x2_img: %f x3_img: %f \n", x1_img[N - 1], x2_img[N - 1], x3_img[N - 1]);*/

	printf("x1[0]: %f, x2[0]: %f , x3[0]: %f , min[0]: %f \n", x1[0], x2[0], x3[0], min[0]);

	printf("streams_avgTime (ms): %f\n", avg / testruns);
	//printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);

	if (debg) {

		memset(errx, 0, 4 * sizeof(float));

		for (int i = 0; i < N; ++i) {

			if (std::fabs(errx[3]) < std::fabs(min[i]) - std::fabs(min_cpu[i])) { errx[3] = std::fabs(min[i]) - std::fabs(min_cpu[i]); erridx = i; }

		}

		std::cout << erridx << std::endl;
		errx[0] = std::fabs(x1[erridx]) - std::fabs(x1_cpu[erridx]);
		errx[1] = std::fabs(x2[erridx]) - std::fabs(x2_cpu[erridx]);
		errx[2] = std::fabs(x3[erridx]) - std::fabs(x3_cpu[erridx]);
		printf("errx1: %f errx2: %f errx3: %f errmin: %f \n", errx[0], errx[1], errx[2], errx[3]);

		std::cout << "checking f(x), (only useful when 3 real roots)" << std::endl;

		printf("f(x_1):  %f \n", A[erridx] * (powf(x1[erridx], 4)) / 4 + B[erridx] * (powf(x1[erridx], 3)) / 3 + C[erridx] * (powf(x1[erridx], 2)) / 2 + D[erridx] * (x1[erridx]) + E[erridx]);
		printf("f(x_3):  %f \n", A[erridx] * (powf(x3[erridx], 4)) / 4 + B[erridx] * (powf(x3[erridx], 3)) / 3 + C[erridx] * (powf(x3[erridx], 2)) / 2 + D[erridx] * (x3[erridx]) + E[erridx]);

	}

	std::cout << "######################################################" << std::endl;
	std::cout << "######################################################" << std::endl;

	printf("Speedup Tcpu/Tgpu: %f \n", avg_cpu / avg);

}