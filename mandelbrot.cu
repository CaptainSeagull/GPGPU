#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#include <assert.h>

#define GPU 1

typedef char unsigned uint8_t;
typedef struct rgb { uint8_t r, g, b; } rgb_t;

static void map_colour(rgb_t *px) {
	if (px->r == 0xFF || px->r == 0) {
		px->r = 0; px->g = 0; px->b = 0;
	} else {
		int const num_shades = 16;
		rgb_t mapping[num_shades] = {
			{ 66,   30,  15 },{ 25,   7,  26 },{ 9,   1,  47 },{ 4,   4,  73 },{ 0,   7, 100 },
			{ 12,   44, 138 },{ 24,  82, 177 },{ 57, 125, 209 },{ 134, 181, 229 },{ 211, 236, 248 },
			{ 241, 233, 191 },{ 248, 201,  95 },{ 255, 170,   0 },{ 204, 128,   0 },{ 153,  87,   0 },
			{ 106,  52,   3 }
		};

		uint8_t uc = px->r % num_shades;
		*px = mapping[uc];
	}
}


__global__ void calculate_mandelbrot(int width, int height, double scale, rgb_t **row_ptrs) {
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
	if (x < zx - 2 * zx * zx + .25)       iter = 0xFF;
	if ((x + 1)*(x + 1) + y * y < 1 / 16) iter = 0xFF;

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
}

__global__ void setRows(rgb_t *img_data, rgb_t **row_ptrs, int height, int width) {
	int i;

	row_ptrs[0] = img_data;
	for (i = 1; i < height; ++i) {
		row_ptrs[i] = row_ptrs[i - 1] + width;
	}
}

#define width (1024)
#define height (1024)

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	rgb_t **cpu_row_ptrs, *cpu_img_data;
	int i, j;
	cudaError_t err;

	//const int width = /*(argc > 1) ? atoi(argv[1]) : */4096 / 4;
	//const int height = /*(argc > 2) ? atoi(argv[2]) : */4096 / 4 ;
	double scale = 1.0 / (width / 4);
	size_t img_data_size, row_ptrs_size;

	//
	// Setup memory.
	//
	{
		// TODO(Jonny): Only do one allocation to save on the overhead.
		
		img_data_size = sizeof(rgb_t) * width *height;
		row_ptrs_size = sizeof(rgb_t *) * height;

		cpu_img_data = (rgb_t *)malloc(img_data_size);
		cpu_row_ptrs = (rgb_t **)malloc(row_ptrs_size);

		cpu_row_ptrs[0] = cpu_img_data;
		for (i = 1; i < height; ++i) {
			cpu_row_ptrs[i] = cpu_row_ptrs[i - 1] + width;
		}
	}

	//
	// Calculate mandelbrot.
	//
	{
		// TODO(Jonny): Can I do the one allocation trick here?
		rgb_t *gpu_img_data, **gpu_row_ptrs;
		int minGridSize, blockSize, dataLength, blocks, grid;

		err = cudaMalloc((void **)&gpu_img_data, img_data_size); assert(err == cudaSuccess);
		err = cudaMalloc((void **)&gpu_row_ptrs, row_ptrs_size); assert(err == cudaSuccess);

		// Set the row information on GPU.
		setRows<<<1, 1>>>(gpu_img_data, gpu_row_ptrs, height, width);

		minGridSize = 0, blockSize = 0;
		dataLength = width * height;
		err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculate_mandelbrot, 0, 0);
		assert(err == cudaSuccess);

		blocks = (int)(pow(2, ceil(log(sqrt(blockSize)) / log(2))));
		grid = (int)((sqrt(dataLength) + blocks - 1) / blocks);

		calculate_mandelbrot << <dim3(grid, grid), dim3(blocks, blocks) >> >(width, height, scale, gpu_row_ptrs);

		err = cudaMemcpy(cpu_img_data, gpu_img_data, sizeof(rgb_t) * width * height, cudaMemcpyDeviceToHost);
		assert(err == cudaSuccess);

		err = cudaFree(gpu_row_ptrs);
		assert(err == cudaSuccess);

		err = cudaFree(gpu_img_data);
		assert(err == cudaSuccess);
	}

	for (i = 0; (i < height); ++i) {
		rgb_t *px = cpu_row_ptrs[i];
		for (j = 0; j < width; j++, px++) {
			map_colour(px);
		}
	}

	//
	// Write to file.
	//
	{
		FILE *fp;

		fp = fopen("out_mandelbrot.ppm", "w");

		fprintf(fp, "P6\n%d %d\n255\n", width, height);
		for (i = height - 1; (i >= 0); --i) {
			fwrite(cpu_row_ptrs[i], 1, width * sizeof(rgb_t), fp);
		}
	}

	//
	// Draw window.
	//
	{
		WNDCLASS wc;
		HWND win;
		HDC hdc;
		bool quit;

		memset(&wc, 0, sizeof(wc));
		wc.lpfnWndProc = DefWindowProc;
		wc.hInstance = hInstance;
		wc.lpszClassName = "Mandelbrot";

		RegisterClass(&wc);

		// Right now, this calculates to the size passed in. Obviously, this is stupid as the user could pass in
		// large sizes...
		win = CreateWindowA(wc.lpszClassName,
			"Mandelbrot",
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT, CW_USEDEFAULT,
			width, height,
			0, 0, hInstance, 0);
		ShowWindow(win, nCmdShow);

		hdc = GetDC(win);

		for (i = 0; i < width; ++i) {
			for (j = 0; j < height; ++j) {
				rgb_t *cur = cpu_row_ptrs[i] + j;
				SetPixel(hdc, i, j, RGB(cur->r, cur->g, cur->b));
			}
		}

		ReleaseDC(win, hdc);

		quit = false;
		while (!quit) {
			MSG msg;
			while (PeekMessageA(&msg, win, 0, 0, PM_REMOVE)) {
				if (msg.message == WM_QUIT) quit = true;

				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
	}

	return(0);
}
