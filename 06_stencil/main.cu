#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

// CPU stencil computation
void cpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);

// GPU stencil computation
__global__ void naive_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);                

int main(int argc, char const *argv[])
{
    // Image size
    int width = 500; int height = 501; int depth = 307;

    // Input image
    float *in_arr = new float[width*height*depth];

    // Output image
    float *out_arr_cpu = new float[width*height*depth];
    float *out_arr_gpu = new float[width*height*depth];
    
    // Initializing images
    srand(time(0));
    for (int i = 0; i < width*height*depth; i++)
    {
        in_arr[i] = static_cast<float>(0 + rand() % 10);
        out_arr_cpu[i] = 0.0f;
    }

    // ======================= //
    // Sequential stencil (CPU)
    // ======================= //
    cpu_stencil(in_arr, out_arr_cpu, width, height, depth);

    // ======================= //
    // Sequential stencil (GPU)
    // ======================= //
    cudaError_t err;
    
    // Memory allocation
    float *d_in_arr;
    err = cudaMalloc((void**) &d_in_arr, width*height*depth*sizeof(float));
    cuda_check(err);

    float *d_out_arr;
    err = cudaMalloc((void**) &d_out_arr, width*height*depth*sizeof(float));
    cuda_check(err);

    // Copy data
    err = cudaMemcpy(d_in_arr, in_arr, width*height*depth*sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err);

    // Stencil computation
    dim3 dim_block(8, 8, 8);
    dim3 dim_grid(ceil(width/8.0), ceil(height/8.0), ceil(depth/8.0));
    naive_gpu_stencil<<<dim_grid, dim_block>>>(d_in_arr, d_out_arr, width, height, depth);

    // Copy result
    err = cudaMemcpy(out_arr_gpu, d_out_arr, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // Assert results
    for (int i = 0; i < width*height*depth; i++)
        assert (out_arr_cpu[i] == out_arr_gpu[i]);
        
    return 0;
}
