#ifndef IMG_BLUR_GPU
# define IMG_BLUR_GPU

void img_blur_kernel(float* P_in, float* P_out, int N, int M, int window);

void img_blur_gpu(float* P_in, float* P_out, int N, int M, int window);

#endif