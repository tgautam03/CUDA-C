CC = nvcc

all: vec_add

vec_add: main.cu vec_add_cpu.c vec_add_gpu.cu ../utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu vec_add_cpu.c vec_add_gpu.cu ../utils.c -o vec_add.out

clean: 
	@echo "Removing object files..."
	rm vec_add.out