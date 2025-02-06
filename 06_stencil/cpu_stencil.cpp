void cpu_stencil(float *in_arr, float *out_arr, int width, int height, int depth)
{
    // Looping over elements of out_arr
    for (int k = 0; k < depth; k++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
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
        }        
    }    
}                