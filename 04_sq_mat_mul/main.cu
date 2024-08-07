#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "sq_mat_mul_cpu.h"
#include "sq_mat_mul_gpu.h"
#include "unco_sq_mat_mul_gpu.h"
#include "../utils.h"

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    int N = 8000;

    // Generate NxN square matrices A and B
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }

    // // Matrix multiplication on a CPU
    // float* C_cpu = (float*)malloc(N*N*sizeof(float));
    // unsigned long long t1_cpu = myCPUTimer();
    // sq_mat_mul_cpu(A, B, C_cpu, N);
    // unsigned long long t2_cpu = myCPUTimer();
    // printf("CPU execution time (N: %d) : %llu microseconds \n", N, t2_cpu-t1_cpu);

    // Coalesced Matrix multiplication on a GPU
    float* C_gpu = (float*)malloc(N*N*sizeof(float));
    unsigned long long t1_gpu = myCPUTimer();
    sq_mat_mul_gpu(A, B, C_gpu, N);
    unsigned long long t2_gpu = myCPUTimer();
    printf("Coalesced GPU execution time (N: %d): %llu microseconds \n", N, t2_gpu-t1_gpu);

    // Uncoalesced Matrix multiplication on a GPU
    float* unco_C_gpu = (float*)malloc(N*N*sizeof(float));
    unsigned long long unco_t1_gpu = myCPUTimer();
    unco_sq_mat_mul_gpu(A, B, unco_C_gpu, N);
    unsigned long long unco_t2_gpu = myCPUTimer();
    printf("Uncoalesced GPU execution time (N: %d): %llu microseconds \n", N, unco_t2_gpu-unco_t1_gpu);

    // Speedup
    printf("\n");
    // printf("Speed-up using Coalesced GPU against CPU (N: %d): %.3f x  \n", N, (double)(t2_cpu-t1_cpu)/(t2_gpu-t1_gpu));
    // printf("Speed-up using Uncoalesced GPU against CPU (N: %d): %.3f x  \n", N, (double)(t2_cpu-t1_cpu)/(unco_t2_gpu-unco_t1_gpu));
    printf("Speed-up using Coalesced GPU against Uncoalesced GPU (N: %d): %.3f x  \n", N, (double)(unco_t2_gpu-unco_t1_gpu)/(t2_gpu-t1_gpu));
    printf("\n");

    // Asserting Results
    printf("Asserting Results... \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // assert(fabs(C_cpu[i*N+j] - C_gpu[i*N+j]) < 0.00000001);
            assert(fabs(C_gpu[i*N+j] - unco_C_gpu[i*N+j]) < 0.00000001);
        }
    }
    printf("Asserting Passed! \n");

    // Free memory
    free(A);
    free(B);
    // free(C_cpu);
    free(C_gpu);
    free(unco_C_gpu);
    
    return 0;
}
