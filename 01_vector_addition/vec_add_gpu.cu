#include <stdio.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

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
    CUDA_CHECK(err_A);
    cudaError_t err_B = cudaMalloc((void**) &d_B, N*sizeof(float));
    CUDA_CHECK(err_B); 
    cudaError_t err_C = cudaMalloc((void**) &d_C, N*sizeof(float));
    CUDA_CHECK(err_C);

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);
    cudaError_t err_B_ = cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);

    // Kernel execution
    vec_add_kernel<<<ceil(N/256.0), 256>>>(d_A, d_B, d_C, N);

    // Copy back results from device to host
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}