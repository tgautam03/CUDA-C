#ifndef C2G_GPU
# define C2G_GPU

void c2g_kernel(float* P_in, float* P_out, int N, int M);

void c2g_gpu(float* P_in, float* P_out, int N, int M);

#endif