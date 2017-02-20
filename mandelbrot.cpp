#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#include <assert.h>

typedef char unsigned uint8_t;
typedef struct rgb { uint8_t r, g, b; } rgb_t;

#define width (1024)
#define height (1024)

struct MandelbrotData {
    int start, thread_cnt;
    double scale;
    rgb_t **row_ptrs;
};

static int const num_shades = 17;
static rgb_t mapping[num_shades] = {
    { 66,   30,  15 }, { 25,   7,   26 }, { 9,     1,  47 }, {   4,   4,  73 }, {   0,   7, 100 },
    { 12,   44, 138 }, { 24,  82,  177 }, { 57,  125, 209 }, { 134, 181, 229 }, { 211, 236, 248 },
    { 241, 233, 191 }, { 248, 201,  95 }, { 255, 170,   0 }, { 204, 128,   0 }, { 153,  87,   0 },
    { 106,  52,   3 }
};

static DWORD
thread_proc(LPVOID param) {
    MandelbrotData *mandelbrot_data = (MandelbrotData *)param;
    int i, j;
    int thread_cnt = mandelbrot_data->thread_cnt;
    double scale = mandelbrot_data->scale;
    rgb_t **row_ptrs = mandelbrot_data->row_ptrs;

    for(i = mandelbrot_data->start; (i < height); i += thread_cnt) {
        for(j = 0; (j < width); ++j) {
            double cx = -.6, cy = 0;
            double y = (i - height / 2) * scale + cy;

            rgb_t *px = row_ptrs[i];
            px += j;

            double x = (j - width / 2) * scale + cx;
            double zx, zy, zx2, zy2;
            uint8_t iter = 0;

            zx = hypot(x - .25, y);
            if(x < zx - 2 * zx * zx + .25)         iter = 0xFF;
            if((x + 1) * (x + 1) + y * y < 1 / 16) iter = 0xFF;

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

            // TODO(Jonny): Is there a way I could add 0x000000 to the mapping so I avoid the if statement?
            if(px->r == 0xFF || px->r == 0) {
                px->r = 0; px->g = 0; px->b = 0;
            } else {
                uint8_t uc = px->r % num_shades;
                *px = mapping[uc];
            }
        }
    }

    return(0);
}

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    rgb_t **row_ptrs, *img_data;
    int i, j;
    void *all_mem;
    int argc;
    char *argv[256];
    char const *cmd = GetCommandLine();
    int cmd_len;
    char *cmdcpy;

    cmd_len = strlen(cmd);
    cmdcpy = (char *)malloc(strlen(cmd) + 1);

    argv[0] = cmdcpy;
    argc = 1;
    for(i = 0; (i < cmd_len); ++i) {
        cmdcpy[i] = cmd[i];
        if(cmdcpy[i] == ' ') {
            cmdcpy[i] = 0;
            argv[argc++] = cmdcpy + (i + 1);
        }
    }
    cmdcpy[cmd_len] = 0;

    // TODO(Jonny): Get the args from GetCommandArgsA or whatever it's called...

    //const int width = /*(argc > 1) ? atoi(argv[1]) : */4096 / 4;
    //const int height = /*(argc > 2) ? atoi(argv[2]) : */4096 / 4 ;
    double scale = 1.0 / (width / 4);
    size_t img_data_size, row_ptrs_size;

    //
    // Setup memory.
    //
    {
        img_data_size = sizeof(rgb_t) * width *height;
        row_ptrs_size = sizeof(rgb_t *) * height;

        all_mem = malloc(img_data_size + row_ptrs_size);

        img_data = (rgb_t *)all_mem;
        row_ptrs = (rgb_t **)((char unsigned *)all_mem + img_data_size);

        row_ptrs[0] = img_data;
        for (i = 1; i < height; ++i) {
            row_ptrs[i] = row_ptrs[i - 1] + width;
        }
    }

    //
    // Calculate mandelbrot.
    //
    {
        MandelbrotData mandelbrot_data[64]; // There won't be as many as 64 processors.
        int thread_cnt;
        SYSTEM_INFO sys_info;

        GetSystemInfo(&sys_info);
        thread_cnt = (sys_info.dwNumberOfProcessors > 0) ? sys_info.dwNumberOfProcessors : 1;

        for(i = 0; (i < thread_cnt); ++i) {
            MandelbrotData *md = mandelbrot_data + i;
            HANDLE hthread;

            md->start = i;
            md->thread_cnt = thread_cnt;
            md->scale = scale;
            md->row_ptrs = row_ptrs;

            hthread = CreateThread(0, 0, thread_proc, md, 0, 0);
            assert((hthread) && (hthread != INVALID_HANDLE_VALUE));
            CloseHandle(hthread);
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
            fwrite(row_ptrs[i], 1, width * sizeof(rgb_t), fp);
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

        for(i = 0; i < width; ++i) {
            for (j = 0; j < height; ++j) {
                rgb_t *cur = row_ptrs[i] + j;
                SetPixel(hdc, i, j, RGB(cur->r, cur->g, cur->b));
            }
        }

        ReleaseDC(win, hdc);

        quit = false;
        while(!quit) {
            MSG msg;
            while (PeekMessageA(&msg, win, 0, 0, PM_REMOVE)) {
                if(msg.message == WM_QUIT) quit = true;

                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
    }

    return(0);
}
