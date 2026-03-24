/*
 * gpu_mem_limit.c — LD_PRELOAD library to enforce per-process GPU memory limits.
 *
 * Intercepts CUDA runtime API calls (cudaMalloc, cudaFree, cudaMemGetInfo)
 * to cap each process at GPU_MEM_LIMIT_MB megabytes of GPU memory.
 *
 * Why runtime API, not driver API?
 *   PyTorch calls cudaMalloc (runtime) through the PLT → LD_PRELOAD works.
 *   libcudart resolves cuMemAlloc_v2 (driver) via internal dlsym → LD_PRELOAD
 *   cannot intercept it without hacking dlsym itself.
 *
 * Usage:
 *   GPU_MEM_LIMIT_MB=88000 LD_PRELOAD=./libgpumemlimit.so python train.py
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -Wall -Wextra -Werror -std=c17 \
 *       -o libgpumemlimit.so gpu_mem_limit.c -ldl -lpthread
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUDA runtime error codes */
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2

/* CUDA stream handle */
typedef void *cudaStream_t;

/* Real function signatures */
typedef cudaError_t (*cudaMalloc_fn)(void **, size_t);
typedef cudaError_t (*cudaFree_fn)(void *);
typedef cudaError_t (*cudaMemGetInfo_fn)(size_t *, size_t *);
typedef cudaError_t (*cudaMallocAsync_fn)(void **, size_t, cudaStream_t);
typedef cudaError_t (*cudaFreeAsync_fn)(void *, cudaStream_t);

/* State */
static size_t g_limit_bytes = 0;   /* 0 = no limit */
static size_t g_used_bytes  = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

/* Real function pointers */
static cudaMalloc_fn      real_cudaMalloc      = NULL;
static cudaFree_fn        real_cudaFree        = NULL;
static cudaMemGetInfo_fn  real_cudaMemGetInfo  = NULL;
static cudaMallocAsync_fn real_cudaMallocAsync = NULL;
static cudaFreeAsync_fn   real_cudaFreeAsync   = NULL;

/* Allocation tracker */
#define MAX_ALLOCS 65536
static struct { void *ptr; size_t size; } g_allocs[MAX_ALLOCS];
static int g_num_allocs = 0;
static int g_dropped = 0;

static void do_init(void) {
    const char *env = getenv("GPU_MEM_LIMIT_MB");
    if (env && atol(env) > 0) {
        g_limit_bytes = (size_t)atol(env) * 1024ULL * 1024ULL;
        fprintf(stderr, "[gpu_mem_limit] limit=%s MB (%zu bytes)\n", env, g_limit_bytes);
    }

    real_cudaMalloc      = (cudaMalloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaFree        = (cudaFree_fn)dlsym(RTLD_NEXT, "cudaFree");
    real_cudaMemGetInfo  = (cudaMemGetInfo_fn)dlsym(RTLD_NEXT, "cudaMemGetInfo");
    real_cudaMallocAsync = (cudaMallocAsync_fn)dlsym(RTLD_NEXT, "cudaMallocAsync");
    real_cudaFreeAsync   = (cudaFreeAsync_fn)dlsym(RTLD_NEXT, "cudaFreeAsync");
}

static void init(void) {
    pthread_once(&g_once, do_init);
}

static void track_alloc(void *ptr, size_t size) {
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

static size_t untrack_alloc(void *ptr) {
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

/* --- Intercepted CUDA runtime functions --- */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    init();
    if (!real_cudaMalloc) return cudaErrorMemoryAllocation;

    pthread_mutex_lock(&g_lock);

    if (g_limit_bytes > 0 && g_used_bytes + size > g_limit_bytes) {
        pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t res = real_cudaMalloc(devPtr, size);
    if (res == cudaSuccess) {
        g_used_bytes += size;
        track_alloc(*devPtr, size);
    }

    pthread_mutex_unlock(&g_lock);
    return res;
}

cudaError_t cudaFree(void *devPtr) {
    init();
    if (!real_cudaFree) return cudaSuccess;

    cudaError_t res = real_cudaFree(devPtr);

    pthread_mutex_lock(&g_lock);
    if (res == cudaSuccess) {
        size_t freed = untrack_alloc(devPtr);
        if (freed > g_used_bytes)
            g_used_bytes = 0;
        else
            g_used_bytes -= freed;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}

cudaError_t cudaMemGetInfo(size_t *free_out, size_t *total_out) {
    init();
    if (!real_cudaMemGetInfo) return 1;

    cudaError_t res = real_cudaMemGetInfo(free_out, total_out);

    pthread_mutex_lock(&g_lock);
    if (res == cudaSuccess && g_limit_bytes > 0) {
        *total_out = g_limit_bytes;
        *free_out  = (g_used_bytes < g_limit_bytes)
                   ? g_limit_bytes - g_used_bytes
                   : 0;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}

/* --- Async variants --- */

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    init();
    if (!real_cudaMallocAsync) return cudaErrorMemoryAllocation;

    pthread_mutex_lock(&g_lock);

    if (g_limit_bytes > 0 && g_used_bytes + size > g_limit_bytes) {
        pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t res = real_cudaMallocAsync(devPtr, size, stream);
    if (res == cudaSuccess) {
        g_used_bytes += size;
        track_alloc(*devPtr, size);
    }

    pthread_mutex_unlock(&g_lock);
    return res;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    init();
    if (!real_cudaFreeAsync) return cudaSuccess;

    cudaError_t res = real_cudaFreeAsync(devPtr, stream);

    pthread_mutex_lock(&g_lock);
    if (res == cudaSuccess) {
        size_t freed = untrack_alloc(devPtr);
        if (freed > g_used_bytes)
            g_used_bytes = 0;
        else
            g_used_bytes -= freed;
    }
    pthread_mutex_unlock(&g_lock);

    return res;
}
