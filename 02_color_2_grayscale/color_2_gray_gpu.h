#ifndef COLOR_2_GRAY_GPU
# define COLOR_2_GRAY_GPU

void color_2_gray_kernel(float* P_in, float* P_out, int N, int M);

void color_2_gray_gpu(float* P_in, float* P_out, int N, int M);

#endif