#define DRAW_WINDOW 0
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <Windows.h>

typedef char unsigned uint8_t;
typedef struct rgb { uint8_t r, g, b; } rgb_t;

__global__ void calculate_mandelbrot(int width, int height, double scale, rgb_t **row_ptrs) {
	int const num_shades = 17;
	rgb_t const mapping[num_shades] = {
		{ 66,   30,  15 },{ 25,   7,   26 },{ 9,     1,  47 },{ 4,   4,  73 },{ 0,   7, 100 },
		{ 12,   44, 138 },{ 24,  82,  177 },{ 57,  125, 209 },{ 134, 181, 229 },{ 211, 236, 248 },
		{ 241, 233, 191 },{ 248, 201,  95 },{ 255, 170,   0 },{ 204, 128,   0 },{ 153,  87,   0 },
		{ 106,  52,   3 }
	};
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	double cx = -.6, cy = 0;
	double y = (i - height / 2) * scale + cy;

	rgb_t *px = row_ptrs[i];
	px += j;

	double x = (j - width / 2) * scale + cx;
	double zx, zy, zx2, zy2;
	uint8_t iter = 0;

	zx = hypot(x - .25, y);
	if (x < zx - 2 * zx * zx + .25)       {iter = 0xFF;}
	if ((x + 1)*(x + 1) + y * y < 1 / 16) {iter = 0xFF;}

	zx = zy = zx2 = zy2 = 0;
	do {
		zy = 2 * zx * zy + y;
		zx = zx2 - zy2 + x;
		zx2 = zx * zx;
		zy2 = zy * zy;
	} while ((iter++ < 0xFF) && (zx2 + zy2 < 4));

	px->r = iter;
	px->g = iter;
	px->b = iter;

	if (px->r == 0xFF || px->r == 0) {
		px->r = 0; px->g = 0; px->b = 0;
	} else {
		uint8_t uc = px->r % num_shades;
		*px = mapping[uc];
	}
}

__global__ void setRows(rgb_t *img_data, rgb_t **row_ptrs, int height, int width) {
	row_ptrs[0] = img_data;
	for (int i = 1; i < height; ++i) {
		row_ptrs[i] = row_ptrs[i - 1] + width;
	}
}

#if DRAW_WINDOW
int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	int const width = 1024;
	int const height = 1024;
#else
int main(int argc, char **argv) {
	int const width = (argc > 1) ? atoi(argv[1]) : 4096 / 4;
	int const height = (argc > 2) ? atoi(argv[2]) : 4096 / 4;
#endif
	rgb_t **cpu_row_ptrs, *cpu_img_data;
	cudaError_t err;

	double scale = 1.0 / (width / 4);
	size_t img_data_size = sizeof(rgb_t) * width *height;
	size_t row_ptrs_size = sizeof(rgb_t *) * height;

	//
	// Setup memory.
	//
	{
		cpu_img_data = (rgb_t *)malloc(img_data_size);
		cpu_row_ptrs = (rgb_t **)malloc(row_ptrs_size);

		cpu_row_ptrs[0] = cpu_img_data;
		for (int i = 1; i < height; ++i) {
			cpu_row_ptrs[i] = cpu_row_ptrs[i - 1] + width;
		}
	}

	//
	// Calculate mandelbrot.
	//
	{
		LARGE_INTEGER cpu_start;
		QueryPerformanceCounter(&cpu_start);

		cudaEvent_t start, stop;
		err = cudaEventCreate(&start);
		assert(err == cudaSuccess);

		err = cudaEventCreate(&stop);
		assert(err == cudaSuccess);

		err = cudaEventRecord(start);
		assert(err == cudaSuccess);

		rgb_t *gpu_img_data, **gpu_row_ptrs;

		err = cudaMalloc((void **)&gpu_img_data, img_data_size);
		assert(err == cudaSuccess);
		
		err = cudaMalloc((void **)&gpu_row_ptrs, row_ptrs_size);
		assert(err == cudaSuccess);

		// Set the row information on GPU.
		setRows << <1, 1 >> >(gpu_img_data, gpu_row_ptrs, height, width);

		int minGridSize = 0, blockSize = 0;
		int dataLength = width * height;
		err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculate_mandelbrot, 0, 0);
		assert(err == cudaSuccess);

		int blocks = (int)(pow(2, ceil(log(sqrt(blockSize)) / log(2))));
		int grid = (int)((sqrt(dataLength) + blocks - 1) / blocks);

		calculate_mandelbrot << <dim3(grid, grid), dim3(blocks, blocks) >> >(width, height, scale, gpu_row_ptrs);

		err = cudaMemcpy(cpu_img_data, gpu_img_data, sizeof(rgb_t) * width * height, cudaMemcpyDeviceToHost);
		assert(err == cudaSuccess);

		err = cudaFree(gpu_row_ptrs);
		assert(err == cudaSuccess);

		err = cudaFree(gpu_img_data);
		assert(err == cudaSuccess);

		err = cudaEventRecord(stop);
		assert(err == cudaSuccess);

		err = cudaEventSynchronize(stop);
		assert(err == cudaSuccess);

		float cuda_ms = 0;
		err = cudaEventElapsedTime(&cuda_ms, start, stop);
		assert(err == cudaSuccess);

		LARGE_INTEGER cpu_end;
		QueryPerformanceCounter(&cpu_end);

		printf("Cuda Timer: %fms\nCPU timer: %llums",
				cuda_ms, (cpu_end.QuadPart - cpu_start.QuadPart) / 1000);
	}

	//
	// Write to file.
	//
	{
		FILE *fp;

		fp = fopen("out_mandelbrot.ppm", "w");

		fprintf(fp, "P6\n%d %d\n255\n", width, height);
		for (int i = height - 1; (i >= 0); --i) {
			fwrite(cpu_row_ptrs[i], 1, width * sizeof(rgb_t), fp);
		}
	}


#if DRAW_WINDOW
	//
	// Draw window.
	//
	{
		WNDCLASS wc;
		memset(&wc, 0, sizeof(wc));
		wc.lpfnWndProc = DefWindowProc;
		wc.hInstance = hInstance;
		wc.lpszClassName = "Mandelbrot";

		RegisterClass(&wc);

		HWND win = CreateWindowA(wc.lpszClassName,
			"Mandelbrot",
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT, CW_USEDEFAULT,
			width, height,
			0, 0, hInstance, 0);
		ShowWindow(win, nCmdShow);

		HDC hdc = GetDC(win);

		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < height; ++j) {
				rgb_t *cur = cpu_row_ptrs[i] + j;
				SetPixel(hdc, i, j, RGB(cur->r, cur->g, cur->b));
			}
		}

		ReleaseDC(win, hdc);

		bool quit = false;
		while (!quit) {
			MSG msg;
			while (PeekMessageA(&msg, win, 0, 0, PM_REMOVE)) {
				if (msg.message == WM_QUIT) quit = true;

				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
	}
#endif

	return(0);
}