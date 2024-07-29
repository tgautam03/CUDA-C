CC = nvcc

all: c2g

c2g: main.cu c2g_cpu.c c2g_gpu.cu ../utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu c2g_cpu.c c2g_gpu.cu ../utils.c -o c2g.out

clean: 
	@echo "Removing object files..."
	rm c2g.out