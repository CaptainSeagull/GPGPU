#define DRAW_WINDOW 0
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <Windows.h>

typedef char unsigned uint8_t;
typedef struct rgb { uint8_t r, g, b; } rgb_t;

struct mandelbrot_data_t {
	int start, thread_cnt;
	double scale;
	rgb_t **row_ptrs;
	int width, height;
};

static int const num_shades = 17;
static rgb_t mapping[num_shades] = {
	{ 66,   30,  15 },{ 25,   7,   26 },{ 9,     1,  47 },{ 4,   4,  73 },{ 0,   7, 100 },
	{ 12,   44, 138 },{ 24,  82,  177 },{ 57,  125, 209 },{ 134, 181, 229 },{ 211, 236, 248 },
	{ 241, 233, 191 },{ 248, 201,  95 },{ 255, 170,   0 },{ 204, 128,   0 },{ 153,  87,   0 },
	{ 106,  52,   3 }
};

static DWORD __stdcall
thread_proc(LPVOID param) {
	mandelbrot_data_t *mandelbrot_data = (mandelbrot_data_t *)param;
	int width = mandelbrot_data->width;
	int height = mandelbrot_data->height;
	int thread_cnt = mandelbrot_data->thread_cnt;
	double scale = mandelbrot_data->scale;
	rgb_t **row_ptrs = mandelbrot_data->row_ptrs;
	double cx = -0.6, cy = 0.0;

	for (int i = mandelbrot_data->start; (i < height); i += thread_cnt) {
		double y = (i - height / 2) * scale + cy;
		for (int j = 0; (j < width); ++j) {
			rgb_t *px = row_ptrs[i] + j;
			double x = (j - width / 2) * scale + cx;
			double zx, zy, zx2, zy2;

			//px += j;

			zx = hypot(x - .25, y);
			uint8_t iter = 0; 
			if (x < zx - 2 * zx * zx + .25)         {iter = 0xFF;}
			if ((x + 1) * (x + 1) + y * y < 1 / 16) {iter = 0xFF;}

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
			}
			else {
				uint8_t uc = px->r % num_shades;
				*px = mapping[uc];
			}
		}
	}

	return(0);
}

#if DRAW_WINDOW
int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	const int width = 1024;
	const int height = 1024;
#else
int main(int argc, char **argv) {
	const int width = (argc > 1) ? atoi(argv[1]) : 4096 / 4;
	const int height = (argc > 2) ? atoi(argv[2]) : 4096 / 4;
#endif
	rgb_t **row_ptrs, *img_data;

	// TODO(Jonny): Get the args from GetCommandArgsA or whatever it's called...

	double scale = 1.0 / (width / 4);
	size_t img_data_size = sizeof(rgb_t) * width * height;
	size_t row_ptrs_size = sizeof(rgb_t *) * height;

	//
	// Setup memory.
	//
	{
		img_data = (rgb_t *)malloc(img_data_size);
		row_ptrs = (rgb_t **)malloc(row_ptrs_size);

		row_ptrs[0] = img_data;
		for (int i = 1; i < height; ++i) {
			row_ptrs[i] = row_ptrs[i - 1] + width;
		}
	}

	//
	// Calculate mandelbrot.
	//
	{
		LARGE_INTEGER start;
		QueryPerformanceCounter(&start);

		int const thread_max = 64;  // There won't be as many as 64 processors.
		mandelbrot_data_t mandelbrot_data[thread_max];
		HANDLE hthreads[thread_max];
		SYSTEM_INFO sys_info;
		GetSystemInfo(&sys_info);
		int thread_cnt = (sys_info.dwNumberOfProcessors > 0) ? sys_info.dwNumberOfProcessors : 1;
		assert(thread_cnt < thread_max);

		for (int i = 0; (i < thread_cnt); ++i) {
			mandelbrot_data_t *md = mandelbrot_data + i;

			md->start = i;
			md->thread_cnt = thread_cnt;
			md->scale = scale;
			md->row_ptrs = row_ptrs;
			md->width = width;
			md->height = height;

			hthreads[i] = CreateThread(0, 0, thread_proc, md, 0, 0);
			assert((hthreads[i] != INVALID_HANDLE_VALUE));
		}

		WaitForMultipleObjects(thread_cnt, hthreads, TRUE, INFINITE);

		LARGE_INTEGER end;
		QueryPerformanceCounter(&end);

		printf("Time: %llums", (end.QuadPart - start.QuadPart) / 1000);
	}

	//
	// Write to file.
	//
	{
		FILE *fp = fopen("out_mandelbrot.ppm", "w");

		fprintf(fp, "P6\n%d %d\n255\n", width, height);
		for (int i = height - 1; (i >= 0); --i) {
			fwrite(row_ptrs[i], 1, width * sizeof(rgb_t), fp);
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
		wc.lpszClassName = L"Mandelbrot";

		RegisterClass(&wc);

		// Right now, this calculates to the size passed in. Obviously, this is stupid as the user could pass in
		// large sizes...
		HWND win = CreateWindow(wc.lpszClassName,
			L"Mandelbrot",
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT, CW_USEDEFAULT,
			width, height,
			0, 0, hInstance, 0);
		assert(win != INVALID_HANDLE_VALUE);
		ShowWindow(win, nCmdShow);

		HDC hdc = GetDC(win);

		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < height; ++j) {
				rgb_t *cur = row_ptrs[i] + j;
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
