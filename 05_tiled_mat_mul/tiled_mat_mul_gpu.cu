#include <stdio.h>
#include <assert.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
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
    for (int phase = 0; phase < ceil((float)N2/TILE_WIDTH); phase++)
    {
        // Load Tiles into shared memory
        if ((i < N1) && ((phase*TILE_WIDTH+tx) < N2))
          sh_A[ty][tx] = A[(i)*N2 + phase*TILE_WIDTH+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < N2) && (j < N3))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N3+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < N1) && (j < N3))
      C[i*N3+j] = value;
}

void tiled_mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3)
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
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(N3/(float)(TILE_WIDTH)), ceil(N1/(float)(TILE_WIDTH)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N1*N3*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}