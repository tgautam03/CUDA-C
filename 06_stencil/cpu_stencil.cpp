void cpu_stencil(float *in_arr, float *out_arr, float stencil[7],
                int width, int height, int depth)
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
                    out_arr[k*width*height + j*width + i] = stencil[0]*in_arr[k*(width*height)+j*width + i]
                                                            + stencil[1]*in_arr[k*(width*height)+j*width + i-1]
                                                            + stencil[2]*in_arr[k*(width*height)+j*width + i+1]
                                                            + stencil[3]*in_arr[k*(width*height)+(j-1)*width + i]
                                                            + stencil[4]*in_arr[k*(width*height)+(j+1)*width + i]
                                                            + stencil[5]*in_arr[(k-1)*(width*height)+j*width + i]
                                                            + stencil[6]*in_arr[(k+1)*(width*height)+j*width + i]; 
                }
            }            
        }        
    }    
}                