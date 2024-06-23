#include <stdio.h>

__global__ void vec_add_kernel(float* A, float* B, float* C, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i] + 0.0f;;
    }
}

void vec_add_gpu(float* A, float* B, float* C, int N)
{
    // Device array pointers
    float* d_A;
    float* d_B;
    float* d_C;

    // Device memory allocation
    cudaError_t err_A = cudaMalloc((void**) &d_A, N*sizeof(float));
    if (err_A != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_A), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    } 
    cudaError_t err_B = cudaMalloc((void**) &d_B, N*sizeof(float));
    if (err_B != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_B), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    } 
    cudaError_t err_C = cudaMalloc((void**) &d_C, N*sizeof(float));
    if (err_C != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_C), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    } 

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    if (err_A_ != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_A_), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    } 
    cudaError_t err_B_ = cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    if (err_B_ != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_B_), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    } 

    // Kernel execution
    vec_add_kernel<<<ceil(N/256.0), 256>>>(d_A, d_B, d_C, N);

    // Copy back results from device to host
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (err_C_ != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err_C_), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}