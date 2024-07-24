#ifndef TILED_SQ_MAT_MUL_GPU
# define TILED_SQ_MAT_MUL_GPU

void tiled_sq_mat_mul_kernel(float* A, float* B, float* C, int N);

void tiled_sq_mat_mul_gpu(float* A, float* B, float* C, int N);

#endif