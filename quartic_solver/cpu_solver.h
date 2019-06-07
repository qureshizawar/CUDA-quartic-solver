#ifndef CPU_SOLVER_H //using instead of #pragma once for greater compatibility
#define CPU_SOLVER_H

void cubicSolver_cpu(int n, float *A, float *B, float *C, float *D, float *Q, float *R, float *del, float *theta, float *sqrtQ,
	float *x1, float *x2, float *x3, float *x1_img, float *x2_img, float *x3_img);

void quarticSolver_cpu(int n, float *A, float *B, float *C, float *D, float *b, float *c, float *d,
	float *Q, float *R, float *Qint, float *Rint, float *del, float *theta, float *sqrtQ, float *x1, float *x2, float *x3, float *temp, float *min);

void quarticSolver_cpu2(int n, float *A, float *B, float *C, float *D, float *b, float *c, float *d,
	float *Q, float *R, float *del, float *theta, float *sqrtQ, float *x1, float *x2, float *x3, float *temp, float *min);

#endif