#ifndef MAT_MUL_GPU
# define MAT_MUL_GPU

void mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3);

#endif