#ifdef __cplusplus
extern "C"
{
#endif

#ifndef MAT_MUL
# define MAT_MUL

void mat_mul_cpu(float* A, float* B, float* C, int N1, int N2, int N3);

#endif

#ifdef __cplusplus
}
#endif