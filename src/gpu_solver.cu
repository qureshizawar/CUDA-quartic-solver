#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>

#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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

__global__
void cubicSolver(int n, float *A, float *B, float *C, float *D, float *Q, float *R, float *del, float *theta, float *sqrtQ,
	float *x1, float *x2, float *x3, float *x1_img, float *x2_img, float *x3_img)
{
	// solver for finding roots (x1, x2, x3) for ax^3 + bx^2 + cx + d = 0

	//__shared__ extern float pis[];
	/*const float twopi = 2 * 3.1415927;
	const float fourpi = 4 * 3.1415927;*/

	//float A[ARRAY_SIZE * THREADBLOCK_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {

		// Q = ((3c/a) - (b^2/a^2)) /9;
		// R = ((9bc)/a^2 - (2b^3)/a^3 - 27d/a) /54;
		// del = Q^3 + R^2;

		Q[i] = ((3 * C[i]) / A[i] - ((B[i] * B[i]) / (A[i] * A[i]))) / 9;
		R[i] = (((-(2 * (B[i] * B[i] * B[i])) / (A[i] * A[i] * A[i]))) + ((9 * (B[i] * C[i])) / (A[i] * A[i])) - ((27 * D[i]) / A[i])) / 54;
		del[i] = ((R[i] * R[i])) + ((Q[i] * Q[i] * Q[i]));

		if (del[i] <= 0) { // all 3 roots real

			/*sqrtQ = 2 * sqrt(-Q);
			theta = acos(R / (sqrtQ ^ 3));

			x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
			x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
			x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

			theta[i] = acosf((float)(R[i] / sqrtf((float)-(Q[i] * Q[i] * Q[i]))));
			sqrtQ[i] = 2 * sqrtf((float)-Q[i]);

			x1[i] = (sqrtQ[i] * cosf((float)(theta[i] / 3)) - (B[i] / (A[i] * 3)));
			x2[i] = (sqrtQ[i] * cosf((float)(theta[i] + 2 * 3.1415927) / 3)) - (B[i] / (A[i] * 3));
			x3[i] = (sqrtQ[i] * cosf((float)(theta[i] + 4 * 3.1415927) / 3)) - (B[i] / (A[i] * 3));
		}

		if (del[i] > 0) { // only 1 real root

			/*S = (R + sqrtD)^(1 / 3);
			T = (R - sqrtD)^(1 / 3);
			x = S + T - b/3;*/

			// real root

			x1[i] = ((cbrtf((float)(R[i] + sqrtf((float)del[i]))))
				+ cbrtf((float)(R[i] - sqrtf((float)del[i])))) - (B[i] / (3 * A[i]));

			x1_img[i] = 0;

			// complex conjugate roots

			x2[i] = -((cbrtf((float)(R[i] + sqrtf((float)del[i])))
				+ cbrtf((float)(R[i] - sqrtf((float)del[i])))) / 2) - (B[i] / (3 * A[i]));

			x2_img[i] = ((sqrtf((float)3) / 2)*(cbrtf((float)(R[i] + sqrtf((float)del[i])))
				- cbrtf((float)(R[i] - sqrtf((float)del[i])))));

			x3[i] = x2[i];

			x3_img[i] = -x2_img[i];

		}

		if (Q[i] == 0 && R[i] == 0) { // all roots real and equal

			x1[i] = -(B[i] / 3);
			x2[i] = x1[i];
			x3[i] = x1[i];

		}

	}
}

__global__
void QRdel(int n, float *A, float *B, float *C, float *D, float *b, float *c, float *d, float *Q, float *R, float *Qint, float *Rint, float *del)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {

		b[i] = 0.75*(B[i] / A[i]);
		c[i] = 0.50*(C[i] / A[i]);
		d[i] = 0.25*(D[i] / A[i]);

		Q[i] = (c[i] / 3.0) - ((b[i] * b[i]) / 9.0);
		R[i] = (b[i] * c[i]) / 6.0 - (b[i] * b[i] * b[i]) / 27.0 - 0.5*d[i];

		// round Q and R to get around problems caused by floating point precision
		Q[i] = roundf(Q[i] * 1E5) / 1E5;
		R[i] = roundf(R[i] * 1E5) / 1E5;

		Qint[i] = (Q[i] * Q[i] * Q[i]);
		Rint[i] = (R[i] * R[i]);

		del[i] = Rint[i] + Qint[i];
		//del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]); // why not just Q*Q*Q + R*R? Heisenbug. Heisenbug in release code
	}
}

// Should be multiple of 32
//#define THREADBLOCK_SIZE 64 
// Could be any number, but the whole array should fit into shared memory 
//#define ARRAY_SIZE 32 

__global__
void QuarticSolver(int n, float *A, float *B, float *C, float *D, float *b, float *Q, float *R, float *del, float *theta, float *sqrtQ,
	float *x1, float *x2, float *x3, float *temp, float *min)
{
	//solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
	//undefined behaviour if A=0

	//__shared__ extern float pis[];
	/*const float twopi = 2 * 3.1415927;
	const float fourpi = 4 * 3.1415927;*/

	//float A[ARRAY_SIZE * THREADBLOCK_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		// comparing against 1E-5 to deal with potential problems with comparing floats to zero
		if (del[i] <= 1E-5) { // all 3 roots real

			/*sqrtQ = 2 * sqrt(-Q);
			theta = acos(R / (sqrtQ ^ 3));

			x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
			x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
			x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

			theta[i] = acosf((float)(R[i] / sqrtf((float)-(Q[i] * Q[i] * Q[i]))));
			sqrtQ[i] = 2 * sqrtf((float)-Q[i]);

			x1[i] = ((sqrtQ[i] * cosf((float)(theta[i]) / 3)) - (b[i] / 3));
			x2[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 2 * 3.1415927) / 3)) - (b[i] / 3));
			x3[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 4 * 3.1415927) / 3)) - (b[i] / 3));

			// unrolled bubble sort  // this vs CUDA sort??
			if (x1[i] < x2[i]) {
				temp[i] = x1[i];
				x1[i] = x2[i];
				x2[i] = temp[i];
			}// { swap(x1[i], x2[i]); }//swap
			if (x2[i] < x3[i]) {
				temp[i] = x2[i];
				x2[i] = x3[i];
				x3[i] = temp[i];
			}//{ swap(x2[i], x3[i]); }//swap
			if (x1[i] < x2[i]) {
				temp[i] = x1[i];
				x1[i] = x2[i];
				x2[i] = temp[i];
			}//{ swap(x1[i], x2[i]); }//swap

			min[i] = A[i] * ((x1[i] * x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i] * x3[i])) / 4 + B[i] * ((x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i])) / 3
				+ C[i] * ((x1[i] * x1[i]) - (x3[i] * x3[i])) / 2 + D[i] * (x1[i] - x3[i])
				<= 0 ? x1[i] : x3[i];

		}

		//if (del[i] > 0) { // only 1 real root
		else {

			/*S = (R + sqrtD)^(1 / 3);
			T = (R - sqrtD)^(1 / 3);
			x = S + T - b/3;*/

			x1[i] = cbrtf((float)(R[i] + sqrtf((float)del[i])))
				+ cbrtf((float)(R[i] - sqrtf((float)del[i]))) - (b[i] / 3); // real root

			// complex conjugate roots not relevant for minimisation

			x2[i] = 0;
			x3[i] = 0;

			min[i] = x1[i];

		}

		// no need as same result as del[i]>0
		/*if (h[i] == 0 && f[i] == 0 && g[i] == 0) { // all roots real and equal

		x1[i] = -b[i] / 3;
		x2[i] = x1[i];
		x3[i] = x1[i];

		min[i] = x1[i];

		}*/

	}
}

__global__
void QuarticSolver_full(int n, float *A, float *B, float *C, float *D, float *b, float *c, float *d, float *Q, float *R, float *del, float *theta, float *sqrtQ,
	float *x1, float *x2, float *x3, float *temp, float *min)
{
	//solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
	//calculates Q, R, del inside the kernel
	//undefined behaviour if A=0

	//__shared__ extern float pis[];
	/*const float twopi = 2 * 3.1415927;
	const float fourpi = 4 * 3.1415927;*/

	//float A[ARRAY_SIZE * THREADBLOCK_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {

		b[i] = 0.75*(B[i] / A[i]);
		c[i] = 0.50*(C[i] / A[i]);
		d[i] = 0.25*(D[i] / A[i]);

		Q[i] = (c[i] / 3.0) - ((b[i] * b[i]) / 9.0);
		R[i] = (b[i] * c[i]) / 6.0 - (b[i] * b[i] * b[i]) / 27.0 - 0.5*d[i];
		del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]);

		if (del[i] <= 0) { // all 3 roots real

			/*sqrtQ = 2 * sqrt(-Q);
			theta = acos(R / (sqrtQ ^ 3));

			x1 = 2 * (sqrtQ*cos(theta / 3) - b/3;
			x2 = 2 * (sqrtQ*cos((theta + 2 * pi) / 3) - b/3);
			x3 = 2 * (sqrtQ*cos((theta + 4 * pi) / 3) - b/3);*/

			theta[i] = acosf((float)(R[i] / sqrtf((float)-(Q[i] * Q[i] * Q[i]))));
			sqrtQ[i] = 2 * sqrtf((float)-Q[i]);

			x1[i] = ((sqrtQ[i] * cosf((float)(theta[i]) / 3)) - (b[i] / 3));
			x2[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 2 * 3.1415927) / 3)) - (b[i] / 3));
			x3[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 4 * 3.1415927) / 3)) - (b[i] / 3));

			// unrolled bubble sort  // this vs CUDA sort??
			if (x1[i] < x2[i]) {
				temp[i] = x1[i];
				x1[i] = x2[i];
				x2[i] = temp[i];
			}// { swap(x1[i], x2[i]); }//swap
			if (x2[i] < x3[i]) {
				temp[i] = x2[i];
				x2[i] = x3[i];
				x3[i] = temp[i];
			}//{ swap(x2[i], x3[i]); }//swap
			if (x1[i] < x2[i]) {
				temp[i] = x1[i];
				x1[i] = x2[i];
				x2[i] = temp[i];
			}//{ swap(x1[i], x2[i]); }//swap

			min[i] = A[i] * ((x1[i] * x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i] * x3[i])) / 4 + B[i] * ((x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i])) / 3
				+ C[i] * ((x1[i] * x1[i]) - (x3[i] * x3[i])) / 2 + D[i] * (x1[i] - x3[i])
				<= 0 ? x1[i] : x3[i];

		}

		if (del[i] > 0) { // only 1 real root

			/*S = (R + sqrtD)^(1 / 3);
			T = (R - sqrtD)^(1 / 3);
			x = S + T - b/3;*/

			x1[i] = cbrtf((float)(R[i] + sqrtf((float)del[i])))
				+ cbrtf((float)(R[i] - sqrtf((float)del[i]))) - (b[i] / 3); // real root

			// complex conjugate roots not relevant for minimisation

			x2[i] = 0;
			x3[i] = 0;

			min[i] = x1[i];

		}

		// no need as same result as del[i]>0
		/*if (h[i] == 0 && f[i] == 0 && g[i] == 0) { // all roots real and equal

		x1[i] = -b[i] / 3;
		x2[i] = x1[i];
		x3[i] = x1[i];

		min[i] = x1[i];

		}*/

	}
}

void QuarticMinimumGPU(int N, float *A, float *B, float *C, float *D, float *E, float *min){
    
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ, *d_Q, *d_R, *d_Qint, *d_Rint, *d_del,
		*d_x1, *d_x2, *d_x3, *d_min, *d_temp;
    
    // kernel dims
	int block_dim = 64;
    
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
    
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, N * sizeof(float), cudaMemcpyHostToDevice);

    //cudaEventRecord(start);

    QRdel << <(N + block_dim - 1) / block_dim, block_dim >> > (N, d_A, d_B, d_C, d_D, d_bi, d_ci, d_di, d_Q, d_R, d_Qint, d_Rint, d_del);

    QuarticSolver << <(N + block_dim - 1) / block_dim, block_dim >> > (N, d_A, d_B, d_C, d_D, d_bi, d_Q, d_R, d_del, d_theta, d_sqrtQ,
        d_x1, d_x2, d_x3, d_temp, d_min);

    //cudaEventRecord(stop);

    /*cudaMemcpy(x1, d_x1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2, d_x2, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x3, d_x3, N * sizeof(float), cudaMemcpyDeviceToHost);*/

    /*cudaMemcpy(de1, d_Q, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(de2, d_R, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(de3, d_del, N * sizeof(float), cudaMemcpyDeviceToHost);*/

    /*cudaMemcpy(x1_img, d_x1_img, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2_img, d_x2_img, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x3_img, d_x3_img, N * sizeof(float), cudaMemcpyDeviceToHost);*/

    cudaMemcpy(min, d_min, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    checkCuda(cudaFree(d_D));
    checkCuda(cudaFree(d_E));
    
    checkCuda(cudaFree(d_bi));
    checkCuda(cudaFree(d_ci));
    checkCuda(cudaFree(d_di));
    checkCuda(cudaFree(d_theta));
    checkCuda(cudaFree(d_sqrtQ));
    
    checkCuda(cudaFree(d_Q));
    checkCuda(cudaFree(d_R));
    checkCuda(cudaFree(d_Qint));
    checkCuda(cudaFree(d_Rint));
    checkCuda(cudaFree(d_del));
    
    checkCuda(cudaFree(d_x1));
    checkCuda(cudaFree(d_x2));
    checkCuda(cudaFree(d_x3));
    
    checkCuda(cudaFree(d_min));
    checkCuda(cudaFree(d_temp));
}

void QuarticMinimumGPUStreams(int N, float *A, float *B, float *C, float *D, float *E, float *min){
    
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_bi, *d_ci, *d_di, *d_theta, *d_sqrtQ, *d_Q, *d_R, *d_Qint, *d_Rint, *d_del,
		*d_x1, *d_x2, *d_x3, *d_min, *d_temp;
    
    // kernel dims
	int block_dim = 64;
    
    // init CUDA streams
	const int nStreams = 4;
	const int streamSize = N / nStreams;
	const int streamBytes = streamSize * sizeof(float);
    
    cudaStream_t stream[nStreams + 1];

	for (int i = 0; i < nStreams; ++i) {
		checkCuda(cudaStreamCreate(&stream[i]));
	}
	checkCuda(cudaStreamCreate(&stream[nStreams]));
    
	int offset = 0;
	const int resstreamSize = N % nStreams;
	const int resstreamBytes = resstreamSize * sizeof(float);
    
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

			/*cudaMemcpyAsync(&x1[offset], &d_x1[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x2[offset], &d_x2[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x3[offset], &d_x3[offset],
				streamBytes, cudaMemcpyDeviceToHost, stream[i]);*/

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

			/*cudaMemcpyAsync(&x1[offset], &d_x1[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
			cudaMemcpyAsync(&x2[offset], &d_x2[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
			cudaMemcpyAsync(&x3[offset], &d_x3[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);*/

			/*cudaMemcpyAsync(&x1_img[offset], &d_x1_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x2_img[offset], &d_x2_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);
			cudaMemcpyAsync(&x3_img[offset], &d_x3_img[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[i]);*/

			cudaMemcpyAsync(&min[offset], &d_min[offset],
			resstreamBytes, cudaMemcpyDeviceToHost, stream[nStreams]);
		}
}