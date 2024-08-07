#ifndef UNCO_SQ_MAT_MUL_GPU
# define UNCO_SQ_MAT_MUL_GPU

void unco_sq_mat_mul_kernel(float* A, float* B, float* C, int N);

void unco_sq_mat_mul_gpu(float* A, float* B, float* C, int N);

#endif