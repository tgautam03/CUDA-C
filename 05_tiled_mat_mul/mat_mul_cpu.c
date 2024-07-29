void mat_mul_cpu(float* A, float* B, float* C, int N1, int N2, int N3)
{
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N3; j++)
        {
            // Value at C[i,j]
            float value = 0;
            for (int k = 0; k < N2; k++)
                value += A[i*N2+k] * B[k*N3+j];

            // Assigning calculated value
            C[i*N3+j] = value;
        }
    }
}