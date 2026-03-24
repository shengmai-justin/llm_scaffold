/*
 * gpu_mem_limit.c — LD_PRELOAD library to enforce per-process GPU memory limits.
 *
 * Intercepts CUDA driver API calls to cap each process at GPU_MEM_LIMIT_MB
 * megabytes of GPU memory. Covers both sync and async allocation paths.
 *
 * Usage:
 *   GPU_MEM_LIMIT_MB=88000 LD_PRELOAD=./libgpumemlimit.so python train.py
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o libgpumemlimit.so gpu_mem_limit.c -ldl -lpthread
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUDA types — avoid requiring CUDA headers to compile */
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUstream;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2

/* Real function signatures */
typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr);
typedef CUresult (*cuMemGetInfo_v2_fn)(size_t *, size_t *);
typedef CUresult (*cuMemAllocAsync_fn)(CUdeviceptr *, size_t, CUstream);
typedef CUresult (*cuMemFreeAsync_fn)(CUdeviceptr, CUstream);

/* State */
static size_t g_limit_bytes = 0;   /* 0 = no limit */
static size_t g_used_bytes  = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

/* Real function pointers */
static cuMemAlloc_v2_fn    real_cuMemAlloc_v2    = NULL;
static cuMemFree_v2_fn     real_cuMemFree_v2     = NULL;
static cuMemGetInfo_v2_fn  real_cuMemGetInfo_v2  = NULL;
static cuMemAllocAsync_fn  real_cuMemAllocAsync  = NULL;
static cuMemFreeAsync_fn   real_cuMemFreeAsync   = NULL;

/* Allocation tracker */
#define MAX_ALLOCS 65536
static struct { CUdeviceptr ptr; size_t size; } g_allocs[MAX_ALLOCS];
static int g_num_allocs = 0;
static int g_dropped = 0;

static void do_init(void) {
    const char *env = getenv("GPU_MEM_LIMIT_MB");
    if (env && atol(env) > 0) {
        g_limit_bytes = (size_t)atol(env) * 1024ULL * 1024ULL;
        fprintf(stderr, "[gpu_mem_limit] limit=%s MB (%zu bytes)\n", env, g_limit_bytes);
    }

    real_cuMemAlloc_v2   = (cuMemAlloc_v2_fn)dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    real_cuMemFree_v2    = (cuMemFree_v2_fn)dlsym(RTLD_NEXT, "cuMemFree_v2");
    real_cuMemGetInfo_v2 = (cuMemGetInfo_v2_fn)dlsym(RTLD_NEXT, "cuMemGetInfo_v2");
    real_cuMemAllocAsync = (cuMemAllocAsync_fn)dlsym(RTLD_NEXT, "cuMemAllocAsync");
    real_cuMemFreeAsync  = (cuMemFreeAsync_fn)dlsym(RTLD_NEXT, "cuMemFreeAsync");
}

static void init(void) {
    pthread_once(&g_once, do_init);
}

static void track_alloc(CUdeviceptr ptr, size_t size) {
    if (g_num_allocs < MAX_ALLOCS) {
        g_allocs[g_num_allocs].ptr  = ptr;
        g_allocs[g_num_allocs].size = size;
        g_num_allocs++;
    } else {
        if (!g_dropped)
            fprintf(stderr, "[gpu_mem_limit] WARNING: alloc tracker full (%d), "
                    "accounting may drift\n", MAX_ALLOCS);
        g_dropped = 1;
    }
}

static size_t untrack_alloc(CUdeviceptr ptr) {
    for (int i = 0; i < g_num_allocs; i++) {
        if (g_allocs[i].ptr == ptr) {
            size_t size = g_allocs[i].size;
            g_allocs[i] = g_allocs[g_num_allocs - 1];
            g_num_allocs--;
            return size;
        }
    }
    return 0;
}

/* Shared alloc/free logic used by both sync and async paths */
static CUresult do_alloc(CUdeviceptr *dptr, size_t bytesize,
                         CUresult (*real_fn)(CUdeviceptr *, size_t)) {
    init();
    if (!real_fn) return CUDA_ERROR_OUT_OF_MEMORY;

    pthread_mutex_lock(&g_lock);

    if (g_limit_bytes > 0 && g_used_bytes + bytesize > g_limit_bytes) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult res = real_fn(dptr, bytesize);
    if (res == CUDA_SUCCESS) {
        g_used_bytes += bytesize;
        track_alloc(*dptr, bytesize);
    }

    pthread_mutex_unlock(&g_lock);
    return res;
}

static CUresult do_free(CUdeviceptr dptr,
                        CUresult (*real_fn)(CUdeviceptr)) {
    init();
    if (!real_fn) return CUDA_SUCCESS;

    CUresult res = real_fn(dptr);

    pthread_mutex_lock(&g_lock);
    if (res == CUDA_SUCCESS) {
        size_t freed = untrack_alloc(dptr);
        if (freed > g_used_bytes)
            g_used_bytes = 0;
        else
            g_used_bytes -= freed;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}

/* --- Intercepted functions: sync --- */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    return do_alloc(dptr, bytesize, real_cuMemAlloc_v2);
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    return do_free(dptr, real_cuMemFree_v2);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

CUresult cuMemGetInfo_v2(size_t *free_out, size_t *total_out) {
    init();
    if (!real_cuMemGetInfo_v2) return 1;

    CUresult res = real_cuMemGetInfo_v2(free_out, total_out);

    pthread_mutex_lock(&g_lock);
    if (res == CUDA_SUCCESS && g_limit_bytes > 0) {
        *total_out = g_limit_bytes;
        *free_out  = (g_used_bytes < g_limit_bytes)
                   ? g_limit_bytes - g_used_bytes
                   : 0;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}

CUresult cuMemGetInfo(size_t *free_out, size_t *total_out) {
    return cuMemGetInfo_v2(free_out, total_out);
}

/* --- Intercepted functions: async (PyTorch cudaMallocAsync path) --- */

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream stream) {
    init();
    if (!real_cuMemAllocAsync) return CUDA_ERROR_OUT_OF_MEMORY;

    pthread_mutex_lock(&g_lock);

    if (g_limit_bytes > 0 && g_used_bytes + bytesize > g_limit_bytes) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult res = real_cuMemAllocAsync(dptr, bytesize, stream);
    if (res == CUDA_SUCCESS) {
        g_used_bytes += bytesize;
        track_alloc(*dptr, bytesize);
    }

    pthread_mutex_unlock(&g_lock);
    return res;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream stream) {
    init();
    if (!real_cuMemFreeAsync) return CUDA_SUCCESS;

    CUresult res = real_cuMemFreeAsync(dptr, stream);

    pthread_mutex_lock(&g_lock);
    if (res == CUDA_SUCCESS) {
        size_t freed = untrack_alloc(dptr);
        if (freed > g_used_bytes)
            g_used_bytes = 0;
        else
            g_used_bytes -= freed;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}
