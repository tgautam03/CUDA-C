CC = nvcc
all: main

main: main.cu cpu_stencil.cpp naive_gpu_stencil.cu tiled_gpu_stencil.cu coarse_tiled_gpu_stencil.cu register_coarse_tiled_gpu_stencil.cu
	@$(CC) cpu_stencil.cpp naive_gpu_stencil.cu tiled_gpu_stencil.cu coarse_tiled_gpu_stencil.cu register_coarse_tiled_gpu_stencil.cu main.cu -o main.out 