#include <stdio.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
    // Working on C[i,j]
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel mat mul
    if (i < N1 && j < N3)
    {
        // Value at C[i,j]
        float value = 0;
        for (int k = 0; k < N2; k++)
        {
            value += A[i*N2+k] * B[k*N3+j];
        }

        // Assigning calculated value
        C[i*N3+j] = value;
    }
}

void mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3)
{
    // Device array pointers
    float* d_A;
    float* d_B;
    float* d_C;

    // Device memory allocation
    cudaError_t err_A = cudaMalloc((void**) &d_A, N1*N2*sizeof(float));
    CUDA_CHECK(err_A);

    cudaError_t err_B = cudaMalloc((void**) &d_B, N2*N3*sizeof(float));
    CUDA_CHECK(err_B);

    cudaError_t err_C = cudaMalloc((void**) &d_C, N1*N3*sizeof(float));
    CUDA_CHECK(err_C);

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N1*N2*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);

    cudaError_t err_B_ = cudaMemcpy(d_B, B, N2*N3*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);

    // Kernel execution
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(N3/32.0), ceil(N1/32.0), 1);
    mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N1*N3*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}