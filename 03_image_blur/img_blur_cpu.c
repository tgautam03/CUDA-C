void img_blur_cpu(float* P_in, float* P_out, int N, int M, int window)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
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
}