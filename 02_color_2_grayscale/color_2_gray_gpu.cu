#include <stdio.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void color_2_gray_kernel(float* P_in, float* P_out, int N, int M)
{
    // Working on P_out[i,j]
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel execution check
    if (i < N && j < M)
    {
        // Position of red color pixel in P_in
        int idx = (i*M+j)*3;

        // Color 2 Gray conversion
        P_out[i*M+j] = 0.21f*P_in[idx] + 0.71f*P_in[idx+1] + 0.07f*P_in[idx+2];
    }
}

void color_2_gray_gpu(float* P_in, float* P_out, int N, int M)
{
    // Device array pointers
    float* d_P_in;
    float* d_P_out;

    // Device memory allocation
    cudaError_t err_in = cudaMalloc((void**) &d_P_in, N*M*3*sizeof(float));
    CUDA_CHECK(err_in);

    cudaError_t err_out = cudaMalloc((void**) &d_P_out, N*M*sizeof(float));
    CUDA_CHECK(err_out);

    // Copying P_in and P_out to device memory
    cudaError_t err_in_ = cudaMemcpy(d_P_in, P_in, N*M*3*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_in_);

    // Kernel execution
    dim3 dim_block(64, 16, 1);
    dim3 dim_grid(ceil(M/64.0), ceil(N/16.0), 1);

    color_2_gray_kernel<<<dim_grid, dim_block>>>(d_P_in, d_P_out, N, M);

    // Copy back results from device to host
    cudaError_t err_out_ = cudaMemcpy(P_out, d_P_out, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_out_);

    // Free device memory
    cudaFree(d_P_in);
    cudaFree(d_P_out);
}