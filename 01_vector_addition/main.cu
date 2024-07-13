#include <stdio.h>
#include <stdlib.h>

#include "vec_add_cpu.h"
#include "vec_add_gpu.h"
#include "../utils.h"

int main(int argc, char const *argv[])
{
    // Length of arrays
    int N = 10000000;
    
    // Memory allocation
    float* A = (float*)malloc(N*sizeof(float));
    float* B = (float*)malloc(N*sizeof(float));
    float* C = (float*)malloc(N*sizeof(float));

    // Initialize A, B and C
    for (int i = 0; i < N; i++)
    {
        A[i] = (float)(rand() % (10 - 0 + 1)+0);
        B[i] = (float)(rand() % (10 - 0 + 1)+0);
        C[i] = 0;
    }

    // Vector addition on a CPU
    unsigned long long t1_cpu = myCPUTimer();
    vec_add_cpu(A, B, C, N);
    unsigned long long t2_cpu = myCPUTimer();
    printf("CPU execution time: %llu microseconds \n", t2_cpu-t1_cpu);
    
    // Printing result
    printf("Array A: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
            printf("%.3f ", A[i]);   
    }
    printf("\n");

    printf("Array B: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
        printf("%.3f ", B[i]);
    }
    printf("\n");

    printf("Array C: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
            printf("%.3f ", C[i]);
    }
    printf("\n");
    printf("\n");

    // Re-initialize C
    for (int i = 0; i < N; i++)
        C[i] = 0;

    // Vector addition on a GPU
    unsigned long long t1_gpu = myCPUTimer();
    vec_add_gpu(A, B, C, N);
    unsigned long long t2_gpu = myCPUTimer();
    printf("GPU execution time: %llu microseconds \n", t2_gpu-t1_gpu);
    
    // Printing result
    printf("Array A: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
            printf("%.3f ", A[i]);   
    }
    printf("\n");

    printf("Array B: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
            printf("%.3f ", B[i]);
    }
    printf("\n");

    printf("Array C: ");
    for (int i = 0; i < 10; i++)
    {
        if (i == 5)
            printf(". . .");
        else
            printf("%.3f ", C[i]);
    }
    printf("\n");


    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
