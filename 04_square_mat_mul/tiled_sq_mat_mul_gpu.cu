#include <stdio.h>
#include <assert.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void tiled_sq_mat_mul_kernel(float* A, float* B, float* C, int N)
{
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Ensure N%TILE_WIDTH == 0
    assert(N % TILE_WIDTH == 0);

    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_WIDTH*by + ty;
    int j = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < N/TILE_WIDTH; phase++)
    {
        // Load Tiles into shared memory
        sh_A[ty][tx] = A[phase*TILE_WIDTH + i*N+tx];
        sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N+j];
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    C[i*N+j] = value;
}

void tiled_sq_mat_mul_gpu(float* A, float* B, float* C, int N)
{
    // Device array pointers
    float* d_A;
    float* d_B;
    float* d_C;

    // Device memory allocation
    cudaError_t err_A = cudaMalloc((void**) &d_A, N*N*sizeof(float));
    CUDA_CHECK(err_A);

    cudaError_t err_B = cudaMalloc((void**) &d_B, N*N*sizeof(float));
    CUDA_CHECK(err_B);

    cudaError_t err_C = cudaMalloc((void**) &d_C, N*N*sizeof(float));
    CUDA_CHECK(err_C);

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);

    cudaError_t err_B_ = cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);

    // Kernel execution
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(N/(float)(TILE_WIDTH)), ceil(N/(float)(TILE_WIDTH)), 1);
    tiled_sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}