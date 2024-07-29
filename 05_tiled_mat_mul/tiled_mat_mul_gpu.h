#ifndef TILED_MAT_MUL_GPU
# define TILED_MAT_MUL_GPU

void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void tiled_mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3);

#endif