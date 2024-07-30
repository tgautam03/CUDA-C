#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mat_mul_cpu.h"
#include "mat_mul_gpu.h"
#include "tiled_mat_mul_gpu.h"
#include "../utils.h"

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    // Matrix A size: N1 x N2
    // Matrix B size: N2 x N3
    int N1 = 1000;
    int N2 = 1000;
    int N3 = 1000;

    // Generate N1xN2 matrix A
    float* A = (float*)malloc(N1*N2*sizeof(float));
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
            A[i*N2+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    // Generate N2xN3 matrix B
    float* B = (float*)malloc(N2*N3*sizeof(float));
    for (int i = 0; i < N2; i++)
    {
        for (int j = 0; j < N3; j++)
            B[i*N3+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    // Matrix multiplication on a CPU
    float* C_cpu = (float*)malloc(N1*N3*sizeof(float));
    unsigned long long t1_cpu = myCPUTimer();
    mat_mul_cpu(A, B, C_cpu, N1, N2, N3);
    unsigned long long t2_cpu = myCPUTimer();
    printf("CPU execution time (N1: %d; N2: %d; N3: %d) : %llu microseconds \n", N1, N2, N3, t2_cpu-t1_cpu);
    printf("\n");

    // Matrix multiplication on a GPU
    float* C_gpu = (float*)malloc(N1*N3*sizeof(float));
    unsigned long long t1_gpu = myCPUTimer();
    mat_mul_gpu(A, B, C_gpu, N1, N2, N3);
    unsigned long long t2_gpu = myCPUTimer();
    printf("GPU execution time (N1: %d; N2: %d; N3: %d): %llu microseconds \n", N1, N2, N3, t2_gpu-t1_gpu);
    printf("\n");

    // Tiled Matrix multiplication on a GPU
    float* C_tiled_gpu = (float*)malloc(N1*N3*sizeof(float));
    unsigned long long t1_tiled_gpu = myCPUTimer();
    tiled_mat_mul_gpu(A, B, C_tiled_gpu, N1, N2, N3);
    unsigned long long t2_tiled_gpu = myCPUTimer();
    printf("Tiled GPU execution time (N1: %d; N2: %d; N3: %d): %llu microseconds \n", N1, N2, N3, t2_tiled_gpu-t1_tiled_gpu);
    printf("\n");

    // Speedup
    printf("Speed-up with GPU (N1: %d; N2: %d; N3: %d): %.3f x  \n", N1, N2, N3, (double)(t2_cpu-t1_cpu)/(t2_gpu-t1_gpu));
    printf("Speed-up with tiled GPU from CPU (N1: %d; N2: %d; N3: %d): %.3f x  \n", N1, N2, N3, (double)(t2_cpu-t1_cpu)/(t2_tiled_gpu-t1_tiled_gpu));
    printf("Speed-up with tiled GPU from untiled GPU (N1: %d; N2: %d; N3: %d): %.3f x  \n", N1, N2, N3, (double)(t2_gpu-t1_gpu)/(t2_tiled_gpu-t1_tiled_gpu));
    printf("\n");

    // Asserting Results
    printf("Asserting Results... \n");
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N3; j++)
        {
            assert(fabs(C_cpu[i*N3+j] - C_gpu[i*N3+j]) < 0.00000001);
            assert(fabs(C_gpu[i*N3+j] - C_tiled_gpu[i*N3+j]) < 0.00000001);
        }
    }
    printf("Asserting Passed! \n");

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    // free(C_gpu);
    
    return 0;
}
