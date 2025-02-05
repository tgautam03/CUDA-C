#include <iostream>
#include <cstdlib>
#include <ctime>

// CPU stencil computation
void cpu_stencil(float *in_arr, float *out_arr, float stencil[7],
                int width, int height, int depth);

int main(int argc, char const *argv[])
{
    // Image size
    int width = 5; int height = 5; int depth = 5;

    // Input image
    float *in_arr = new float[width*height*depth];

    // Output image
    float *out_arr = new float[width*height*depth];

    // Stencil
    float stencil[7] = {1, 1, 1, 1, 1, 1, 1};
    
    // Initializing images
    srand(time(0));
    for (int i = 0; i < width*height*depth; i++)
    {
        in_arr[i] = static_cast<float>(0 + rand() % 10);
        out_arr[i] = 0.0f;
    }

    // Sequential stencil (CPU)
    cpu_stencil(in_arr, out_arr, stencil, width, height, depth);

    // Print output
    for (int k = 0; k < depth; k++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                std::cout << out_arr[k*width*height + j*width + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    

    return 0;
}
