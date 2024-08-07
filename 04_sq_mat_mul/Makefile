CC = nvcc

all: sq_mat_mul

sq_mat_mul: main.cu sq_mat_mul_cpu.c sq_mat_mul_gpu.cu unco_sq_mat_mul_gpu.cu ../utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu sq_mat_mul_cpu.c sq_mat_mul_gpu.cu unco_sq_mat_mul_gpu.cu ../utils.c -o sq_mat_mul.out

clean: 
	@echo "Removing object files..."
	rm sq_mat_mul.out