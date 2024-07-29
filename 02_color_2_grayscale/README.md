# Color to Grayscale Conversion

## Problem Statement
### Inputs
- `N x M X 3`: Image Matrix Dimension
- `P_in`: Input Image
- `N x M`: Output Matrix Dimension
- `P_out_cpu`: Output Image from CPU computation
- `P_out_cpu`: Output Image from GPU computation

### Operation
`P_out[:,:] = 0.21f*P_in[:,:,0] + 0.71f*P_in[:,:,1] + 0.07f*P_in[:,:,2];`

### Compilation and execution
To compile run `make` in the terminal. This is create an executable `c2g.out`. 