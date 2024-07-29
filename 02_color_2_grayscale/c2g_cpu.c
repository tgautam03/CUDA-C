void c2g_cpu(float* P_in, float* P_out, int N, int M)
{
    for (int i = 0; i < N*M; i++)
    {
        // Position of red color pixel in P_in
        int idx = i*3;

        // Color 2 Gray conversion
        P_out[i] = 0.21f*P_in[idx] + 0.71f*P_in[idx+1] + 0.07f*P_in[idx+2];
    }
}