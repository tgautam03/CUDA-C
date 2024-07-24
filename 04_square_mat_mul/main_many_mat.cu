#include <stdio.h>
#include <stdlib.h>

#include "sq_mat_mul_cpu.h"
#include "sq_mat_mul_gpu.h"
#include "../utils.h"

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    long long unsigned cpu_times[15];
    long long unsigned gpu_times[15];
    int mat_sizes[15];
    int count = 0;
    for (int N = 1000; N  < 8001; N+=500)
    {
        mat_sizes[count] = N;

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
        
        // printf("Printing A: \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         // Position of red color pixel in P_in
        //         printf("%.3f ", A[i*N+j]);
        //     }
        //     printf("\n");
        // }

        // printf("Printing B: \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         // Position of red color pixel in P_in
        //         printf("%.3f ", B[i*N+j]);
        //     }
        //     printf("\n");
        // }

        // Matrix multiplication on a CPU
        float* C_cpu = (float*)malloc(N*N*sizeof(float));
        unsigned long long t1_cpu = myCPUTimer();
        sq_mat_mul_cpu(A, B, C_cpu, N);
        unsigned long long t2_cpu = myCPUTimer();
        printf("CPU execution time (N: %d) : %llu microseconds \n", N, t2_cpu-t1_cpu);

        // printf("Printing C (CPU): \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         // Position of red color pixel in P_in
        //         printf("%.3f ", C_cpu[i*N+j]);
        //     }
        //     printf("\n");
        // }

        // Matrix multiplication on a GPU
        float* C_gpu = (float*)malloc(N*N*sizeof(float));
        unsigned long long t1_gpu = myCPUTimer();
        sq_mat_mul_gpu(A, B, C_gpu, N);
        unsigned long long t2_gpu = myCPUTimer();
        printf("GPU execution time (N: %d): %llu microseconds \n", N, t2_gpu-t1_gpu);

        // Speedup
        printf("Speed-up from GPU (N: %d): %.3f x  \n", N, (double)(t2_cpu-t1_cpu)/(t2_gpu-t1_gpu));
        printf("\n");

        // printf("Printing C (GPU): \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         // Position of red color pixel in P_in
        //         printf("%.3f ", C_gpu[i*N+j]);
        //     }
        //     printf("\n");
        // }

        cpu_times[count] = t2_cpu-t1_cpu;
        gpu_times[count] = t2_gpu-t1_gpu;
        count += 1;

        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    printf("Matrix Sizes: ");
    for (int i = 0; i < count-1; i++)
        printf("%d ", mat_sizes[i]);
    printf("\n");

    printf("CPU Times: ");
    for (int i = 0; i < count-1; i++)
        printf("%llu ", cpu_times[i]);
    printf("\n");
    
    printf("GPU Times: ");
    for (int i = 0; i < count-1; i++)
        printf("%llu ", gpu_times[i]);
    printf("\n");
    
    
    return 0;
}
