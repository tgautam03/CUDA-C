void sq_mat_mul_cpu(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Value at C[i,j]
            float value = 0;
            for (int k = 0; k < N; k++)
            {
                value += A[i*N+k] * B[k*N+j];
            }

            // Assigning calculated value
            C[i*N+j] = value;
        }
    }
}