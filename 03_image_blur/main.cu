#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "img_blur_cpu.h"
#include "img_blur_gpu.h"

#define MAX_NUM 10 
#define MIN_NUM 0 

int main(int argc, char const *argv[])
{
    // Generate a dummy NxM grayscale image
    int N = 5;
    int M = 10;
    float* P_in = (float*)malloc(N*M*sizeof(float));
    for (int i = 0; i < N*M; i++)
        P_in[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    
    printf("Printing original image: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%.3f ", P_in[i]);
        }
        printf("\n");
    }
    
    // Bluring the grayscale image on a CPU
    float* P_out_cpu = (float*)malloc(N*M*sizeof(float));
    img_blur_cpu(P_in, P_out_cpu, N, M, 1);
    
    printf("Printing img_blur_cpu: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%.3f ", P_out_cpu[i*M+j]);
        }
        printf("\n");
    }

    // Bluring the grayscale image on a GPU
    float* P_out_gpu = (float*)malloc(N*M*sizeof(float));
    img_blur_gpu(P_in, P_out_gpu, N, M, 1);
    
    printf("Printing img_blur_gpu: \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%.3f ", P_out_gpu[i*M+j]);
        }
        printf("\n");
    }

    // Asserting Results
    printf("\n");
    printf("Asserting Results... \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
            assert(fabs(P_out_cpu[i*M+j] - P_out_gpu[i*M+j]) < 0.00000001);
    }
    printf("Asserting Passed! \n");

    
    return 0;
}
