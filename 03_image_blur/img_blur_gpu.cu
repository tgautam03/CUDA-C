#include <stdio.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void img_blur_kernel(float* P_in, float* P_out, int N, int M, int window)
{
    // Working on P_out[i,j]
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel execution check
    if (i < N && j < M)
    {
        // Looping over window width in x and y dims
        float value = 0;
        float num_pixels = 0;
        for (int k = -window; k < window+1; k++)
        {
            for (int l = -window; l < window+1; l++)
            {
                if ((i+k) >= 0 && (i+k) < N && (j+l) >= 0 && (j+l) < M)
                {
                    value += P_in[(i+k)*M+(j+l)];
                    num_pixels+=1;
                }
            }
        }
        P_out[i*M+j] = value/num_pixels;
    }
}

void img_blur_gpu(float* P_in, float* P_out, int N, int M, int window)
{
    // Device array pointers
    float* d_P_in;
    float* d_P_out;

    // Device memory allocation
    cudaError_t err_in = cudaMalloc((void**) &d_P_in, N*M*sizeof(float));
    CUDA_CHECK(err_in);

    cudaError_t err_out = cudaMalloc((void**) &d_P_out, N*M*sizeof(float));
    CUDA_CHECK(err_out);

    // Copying P_in and P_out to device memory
    cudaError_t err_in_ = cudaMemcpy(d_P_in, P_in, N*M*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_in_);

    // Kernel execution
    dim3 dim_block(64, 16, 1);
    dim3 dim_grid(ceil(M/64.0), ceil(N/16.0), 1);

    img_blur_kernel<<<dim_grid, dim_block>>>(d_P_in, d_P_out, N, M, window);

    // Copy back results from device to host
    cudaError_t err_out_ = cudaMemcpy(P_out, d_P_out, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_out_);

    // Free device memory
    cudaFree(d_P_in);
    cudaFree(d_P_out);
}