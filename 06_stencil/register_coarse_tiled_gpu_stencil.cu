#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void register_coarse_tiled_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth)
{
    // Mapping threads to elements of out_arr
    const int k_start = blockIdx.z*OUT_TILE_DIM;
    const int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    const int i = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    // Shared memory allocation 
    float in_prev;
    __shared__ float in_curr_sh_tile[IN_TILE_DIM][IN_TILE_DIM];
    float in_curr;
    float in_next;

    // Moving data to shared memory
    if ((i >= 0) && (i < width) && (j >= 0) && (j < height) && (k_start-1 >= 0) && (k_start-1 < depth))
        in_prev = in_arr[(k_start-1)*(width*height)+j*(width) + i];
    else
        in_prev = 0.0f;

	if ((i >= 0) && (i < width) && (j >= 0) && (j < height) && (k_start >= 0) && (k_start < depth))
	{
		in_curr = in_arr[k_start*(width*height)+j*(width) + i];
		in_curr_sh_tile[threadIdx.y][threadIdx.x] = in_curr;
		
	}
    else
    {
	    in_curr = 0.0f;
	    in_curr_sh_tile[threadIdx.y][threadIdx.x] = 0.0f;
	}

	for(int k = k_start; k < k_start+OUT_TILE_DIM; k++)
	{
		if ((i >= 0) && (i < width) && (j >= 0) && (j < height) && (k+1 >= 0) && (k+1 < depth))
	        in_next = in_arr[(k+1)*(width*height)+j*(width) + i];
	    else
	        in_next = 0.0f;

		// Ensure that all tiles are loaded before proceeding
		__syncthreads();
    
	    // Stencil computation
	    if ((k > 0 && k < depth-1) && (j > 0 && j < height-1) && (i > 0 && i < width-1))
	    {
	        if ((threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM-1) && (threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM-1))
	        {
	            out_arr[k*width*height + j*width + i] = 1*in_curr
	                                                    + 1*in_curr_sh_tile[threadIdx.y][threadIdx.x-1]
	                                                    + 1*in_curr_sh_tile[threadIdx.y][threadIdx.x+1]
	                                                    + 1*in_curr_sh_tile[threadIdx.y-1][threadIdx.x]
	                                                    + 1*in_curr_sh_tile[threadIdx.y+1][threadIdx.x]
	                                                    + 1*in_prev
	                                                    + 1*in_next; 
	        }
	    }
	    __syncthreads();

		in_prev = in_curr;
		in_curr = in_next;
		in_curr_sh_tile[threadIdx.y][threadIdx.x] = in_next;
	}  
}   