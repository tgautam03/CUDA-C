#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "c2g_cpu.h"
#include "c2g_gpu.h"

#define MAX_NUM 255 
#define MIN_NUM 0 

int main(int argc, char const *argv[])
{
    // Generate a dummy NxM RGB image
    int N = 2;
    int M = 4;
    float* P_in = (float*)malloc(N*M*3*sizeof(float));
    for (int i = 0; i < N*M*3; i++)
        P_in[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        
    
    // printf("Printing original image: \n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < M; j++)
    //     {
    //         // Position of red color pixel in P_in
    //         int idx = (i*M+j)*3;
    //         printf("(%.3f %.3f %.3f) ", P_in[idx], P_in[idx+1], P_in[idx+2]);
    //     }
    //     printf("\n");
    // }
    
    // Converting to grayscale on a CPU
    float* P_out_cpu = (float*)malloc(N*M*sizeof(float));
    c2g_cpu(P_in, P_out_cpu, N, M);
    
    printf("Printing c2g_cpu: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%.3f ", P_out_cpu[i*M+j]);
        }
        printf("\n");
    }

    // Converting to grayscale on a GPU
    float* P_out_gpu = (float*)malloc(N*M*sizeof(float));
    c2g_gpu(P_in, P_out_gpu, N, M);
    
    printf("Printing c2g_gpu: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%.3f ", P_out_gpu[i*M+j]);
        }
        printf("\n");
    }

    return 0;
}
