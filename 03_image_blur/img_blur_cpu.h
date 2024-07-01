#ifdef __cplusplus
extern "C"
{
#endif

#ifndef IMG_BLUR_CPU
# define IMG_BLUR_CPU

void img_blur_cpu(float* P_in, float* P_out, int N, int M, int window);

#endif

#ifdef __cplusplus
}
#endif