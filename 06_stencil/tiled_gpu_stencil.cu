#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void tiled_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth)
{
    // Mapping threads to elements of out_arr
    const int k = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    const int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    const int i = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    // Shared memory allocation 
    __shared__ float in_sh_tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Moving data to shared memory
    if ((i >= 0) && (i < width) && (j >= 0) && (j < height) && (k >= 0) && (k < depth))
        in_sh_tile[threadIdx.z][threadIdx.y][threadIdx.x] = in_arr[k*(width*height)+j*(width) + i];
    else
        in_sh_tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    
    // Ensure that all tiles are loaded before proceeding
	__syncthreads();
    
    // Stencil computation
    if ((k > 0 && k < depth-1) && (j > 0 && j < height-1) && (i > 0 && i < width-1))
    {
        if ((threadIdx.z > 0 && threadIdx.z < IN_TILE_DIM-1) && (threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM-1) && (threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM-1))
        {
            out_arr[k*width*height + j*width + i] = 1*in_sh_tile[threadIdx.z][threadIdx.y][threadIdx.x]
                                                    + 1*in_sh_tile[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                                    + 1*in_sh_tile[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                                    + 1*in_sh_tile[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                                    + 1*in_sh_tile[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                                    + 1*in_sh_tile[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                                    + 1*in_sh_tile[threadIdx.z+1][threadIdx.y][threadIdx.x]; 
        }
    }    
}                