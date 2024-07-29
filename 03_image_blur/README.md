# Image Blur

## Problem Statement
### Inputs
- `N x M`: Image Matrix Dimension
- `P_in`: Input Image
- `N x M`: Output Matrix Dimension
- `P_out_cpu`: Output Image from CPU computation
- `P_out_cpu`: Output Image from GPU computation

### Operation
Windowed averaging over all pixels.

### Compilation and execution
To compile run `make` in the terminal. This creates an executable `img_blur.out`. 