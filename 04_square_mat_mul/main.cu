#include <stdio.h>
#include <stdlib.h>

#include "sq_mat_mul_cpu.h"
#include "sq_mat_mul_gpu.h"

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    // Generate NxN square matrices A and B
    int N = 2;
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
    
    printf("Printing A: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Position of red color pixel in P_in
            printf("%.3f ", A[i*N+j]);
        }
        printf("\n");
    }

    printf("Printing B: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Position of red color pixel in P_in
            printf("%.3f ", B[i*N+j]);
        }
        printf("\n");
    }

    // Matrix multiplication on a CPU
    float* C_cpu = (float*)malloc(N*N*sizeof(float));
    sq_mat_mul_cpu(A, B, C_cpu, N);

    printf("Printing C (CPU): \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Position of red color pixel in P_in
            printf("%.3f ", C_cpu[i*N+j]);
        }
        printf("\n");
    }

    // Matrix multiplication on a GPU
    float* C_gpu = (float*)malloc(N*N*sizeof(float));
    sq_mat_mul_gpu(A, B, C_gpu, N);

    printf("Printing C (GPU): \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Position of red color pixel in P_in
            printf("%.3f ", C_gpu[i*N+j]);
        }
        printf("\n");
    }
    
    return 0;
}
