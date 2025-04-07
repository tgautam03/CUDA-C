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

// GPU naive stencil computation
__global__ void naive_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);   

// GPU tiled stencil computation
#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__global__ void tiled_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);

// GPU coarse tiled stencil computation
#define IN_COARSE_TILE_DIM 32
#define OUT_COARSE_TILE_DIM (IN_COARSE_TILE_DIM - 2)
__global__ void coarse_tiled_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);

// GPU register coarse tiled stencil computation
__global__ void register_coarse_tiled_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth);

int main(int argc, char const *argv[])
{
    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Image size
    unsigned int width = 501; unsigned int height = 711; unsigned int depth = 111;

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

    // ======================== //
    // Sequential stencil (CPU) //
    // ======================== //
    // Warmup runs (not necessary for CPU)
    for (int i = 0; i < 10; i++)
        cpu_stencil(in_arr, out_arr_cpu, width, height, depth);

    cudaEventRecord(beg);
    for (int i = 0; i < 10; i++)
        cpu_stencil(in_arr, out_arr_cpu, width, height, depth);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    std::cout << "CPU runtime: " << elapsed_time / 10.0 << " secs \n";

    // ========= //
    // GPU setup //
    // ========= //
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

    // ============================ //
    // Naive parallel stencil (GPU) //
    // ============================ //
    dim3 dim_block(8, 8, 8);
    dim3 dim_grid(ceil(width/8.0), ceil(height/8.0), ceil(depth/8.0));
    
    // Warmup runs
    for (int i = 0; i < 10; i++)
        naive_gpu_stencil<<<dim_grid, dim_block>>>(d_in_arr, d_out_arr, width, height, depth);

    cudaEventRecord(beg);
    for (int i = 0; i < 10; i++)
        naive_gpu_stencil<<<dim_grid, dim_block>>>(d_in_arr, d_out_arr, width, height, depth);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    std::cout << "GPU (naive) runtime: " << elapsed_time / 10.0 << " secs \n";

    // Copy result
    err = cudaMemcpy(out_arr_gpu, d_out_arr, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // Assert results
    for (int i = 0; i < width*height*depth; i++)
        assert (out_arr_cpu[i] == out_arr_gpu[i]);

    // ============================ //
    // Tiled parallel stencil (GPU) //
    // ============================ //
    dim3 dim_block_tiled(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 dim_grid_tiled(ceil(width/static_cast<float>(OUT_TILE_DIM)), ceil(height/static_cast<float>(OUT_TILE_DIM)), ceil(depth/static_cast<float>(OUT_TILE_DIM)));
    
    // Warmup runs
    for (int i = 0; i < 10; i++)
        tiled_gpu_stencil<<<dim_grid_tiled, dim_block_tiled>>>(d_in_arr, d_out_arr, width, height, depth);

    cudaEventRecord(beg);
    for (int i = 0; i < 10; i++)
        tiled_gpu_stencil<<<dim_grid_tiled, dim_block_tiled>>>(d_in_arr, d_out_arr, width, height, depth);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    std::cout << "GPU (tiled) runtime: " << elapsed_time / 10.0 << " secs \n";

    // Copy result
    err = cudaMemcpy(out_arr_gpu, d_out_arr, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // Assert results
    for (int i = 0; i < width*height*depth; i++)
        assert (out_arr_cpu[i] == out_arr_gpu[i]);
        
    // =================================== //
    // Coarse Tiled parallel stencil (GPU) //
    // =================================== //
    dim3 dim_block_coarse_tiled(IN_COARSE_TILE_DIM, IN_COARSE_TILE_DIM, IN_COARSE_TILE_DIM);
    dim3 dim_grid_coarse_tiled(ceil(width/static_cast<float>(OUT_COARSE_TILE_DIM)), ceil(height/static_cast<float>(OUT_COARSE_TILE_DIM)), ceil(depth/static_cast<float>(OUT_COARSE_TILE_DIM)));
    
    // Warmup runs
    for (int i = 0; i < 10; i++)
        coarse_tiled_gpu_stencil<<<dim_grid_coarse_tiled, dim_block_coarse_tiled>>>(d_in_arr, d_out_arr, width, height, depth);

    cudaEventRecord(beg);
    for (int i = 0; i < 10; i++)
        coarse_tiled_gpu_stencil<<<dim_grid_coarse_tiled, dim_block_coarse_tiled>>>(d_in_arr, d_out_arr, width, height, depth);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    std::cout << "GPU (coarse tiled) runtime: " << elapsed_time / 10.0 << " secs \n";

    // Copy result
    err = cudaMemcpy(out_arr_gpu, d_out_arr, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // Assert results
    for (int i = 0; i < width*height*depth; i++)
        assert (out_arr_cpu[i] == out_arr_gpu[i]);

    // ============================================ //
    // Register coarse Tiled parallel stencil (GPU) //
    // ============================================ //
    dim3 dim_block_register_coarse_tiled(IN_COARSE_TILE_DIM, IN_COARSE_TILE_DIM, IN_COARSE_TILE_DIM);
    dim3 dim_grid_register_coarse_tiled(ceil(width/static_cast<float>(OUT_COARSE_TILE_DIM)), ceil(height/static_cast<float>(OUT_COARSE_TILE_DIM)), ceil(depth/static_cast<float>(OUT_COARSE_TILE_DIM)));
    
    // Warmup runs
    for (int i = 0; i < 10; i++)
        register_coarse_tiled_gpu_stencil<<<dim_grid_register_coarse_tiled, dim_block_register_coarse_tiled>>>(d_in_arr, d_out_arr, width, height, depth);

    cudaEventRecord(beg);
    for (int i = 0; i < 10; i++)
        register_coarse_tiled_gpu_stencil<<<dim_grid_register_coarse_tiled, dim_block_register_coarse_tiled>>>(d_in_arr, d_out_arr, width, height, depth);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    std::cout << "GPU (register coarse tiled) runtime: " << elapsed_time / 10.0 << " secs \n";

    // Copy result
    err = cudaMemcpy(out_arr_gpu, d_out_arr, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);

    // Assert results
    for (int i = 0; i < width*height*depth; i++)
        assert (out_arr_cpu[i] == out_arr_gpu[i]);

    return 0;
}
