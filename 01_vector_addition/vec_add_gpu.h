#ifndef VEC_ADD_GPU
# define VEC_ADD_GPU

void vec_add_kernel(float* A, float* B, float* C, int N);

void vec_add_gpu(float* A, float* B, float* C, int N);

#endif