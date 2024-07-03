#ifndef SQ_MAT_MUL_GPU
# define SQ_MAT_MUL_GPU

void sq_mat_mul_kernel(float* A, float* B, float* C, int N);

void sq_mat_mul_gpu(float* A, float* B, float* C, int N);

#endif