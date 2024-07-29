CC = nvcc

all: mat_mul

mat_mul: main.cu mat_mul_cpu.c mat_mul_gpu.cu tiled_mat_mul_gpu.cu ../utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu mat_mul_cpu.c mat_mul_gpu.cu tiled_mat_mul_gpu.cu ../utils.c -o mat_mul.out

clean: 
	@echo "Removing object files..."
	rm mat_mul.out