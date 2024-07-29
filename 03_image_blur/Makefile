CC = nvcc

all: img_blur

img_blur: main.cu img_blur_cpu.c img_blur_gpu.cu ../utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu img_blur_cpu.c img_blur_gpu.cu ../utils.c -o img_blur.out

clean: 
	@echo "Removing object files..."
	rm img_blur.out