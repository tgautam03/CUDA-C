#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SQ_MAT_MUL
# define SQ_MAT_MUL

void sq_mat_mul_cpu(float* A, float* B, float* C, int N);

#endif

#ifdef __cplusplus
}
#endif