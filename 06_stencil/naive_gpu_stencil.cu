__global__ void naive_gpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth)
{
    // Mapping threads to elements of out_arr
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ((k > 0 && k < depth-1) && (j > 0 && j < height-1) && (i > 0 && i < width-1))
    {
        out_arr[k*width*height + j*width + i] = 1*in_arr[k*(width*height)+j*width + i]
                                                + 1*in_arr[k*(width*height)+j*width + i-1]
                                                + 1*in_arr[k*(width*height)+j*width + i+1]
                                                + 1*in_arr[k*(width*height)+(j-1)*width + i]
                                                + 1*in_arr[k*(width*height)+(j+1)*width + i]
                                                + 1*in_arr[(k-1)*(width*height)+j*width + i]
                                                + 1*in_arr[(k+1)*(width*height)+j*width + i]; 
    }    
}                