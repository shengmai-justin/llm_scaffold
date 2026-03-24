/*
 * gpu_mem_limit.c — LD_PRELOAD library to enforce per-process GPU memory limits.
 *
 * Intercepts CUDA driver API calls to cap each process at GPU_MEM_LIMIT_MB
 * megabytes of GPU memory. Covers both sync and async allocation paths.
 *
 * Key insight: CUDA runtime (libcudart) resolves driver API symbols via
 * dlsym(libcuda_handle, "cuMemAlloc_v2"), which bypasses normal LD_PRELOAD.
 * We intercept dlsym itself to redirect those lookups to our wrappers.
 *
 * Usage:
 *   GPU_MEM_LIMIT_MB=88000 LD_PRELOAD=./libgpumemlimit.so python train.py
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -Wall -Wextra -Werror -std=c17 \
 *       -o libgpumemlimit.so gpu_mem_limit.c -ldl -lpthread
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUDA types — avoid requiring CUDA headers to compile */
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUstream;

/* dlsym constants — defined manually to avoid including <dlfcn.h>
 * which declares dlsym with __THROW/__nonnull attributes that
 * conflict with our override definition under -Werror */
#define RTLD_NEXT ((void *)-1)

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

/* Real function pointers (resolved via __libc_dlsym to avoid recursion) */
static cuMemAlloc_v2_fn    real_cuMemAlloc_v2    = NULL;
static cuMemFree_v2_fn     real_cuMemFree_v2     = NULL;
static cuMemGetInfo_v2_fn  real_cuMemGetInfo_v2  = NULL;
static cuMemAllocAsync_fn  real_cuMemAllocAsync  = NULL;
static cuMemFreeAsync_fn   real_cuMemFreeAsync   = NULL;

/* glibc internal dlsym — needed to resolve the real dlsym without recursion */
extern void *__libc_dlsym(void *handle, const char *name);

/* Real dlsym pointer */
typedef void *(*dlsym_fn)(void *, const char *);
static dlsym_fn real_dlsym = NULL;

/* Allocation tracker */
#define MAX_ALLOCS 65536
static struct { CUdeviceptr ptr; size_t size; } g_allocs[MAX_ALLOCS];
static int g_num_allocs = 0;
static int g_dropped = 0;

/* Forward declarations of our intercepted CUDA functions */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemGetInfo_v2(size_t *free_out, size_t *total_out);
CUresult cuMemGetInfo(size_t *free_out, size_t *total_out);
CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream stream);
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream stream);

static void do_init(void) {
    const char *env = getenv("GPU_MEM_LIMIT_MB");
    if (env && atol(env) > 0) {
        g_limit_bytes = (size_t)atol(env) * 1024ULL * 1024ULL;
        fprintf(stderr, "[gpu_mem_limit] limit=%s MB (%zu bytes)\n", env, g_limit_bytes);
    }

    /* Resolve real dlsym via glibc internal */
    real_dlsym = (dlsym_fn)__libc_dlsym(RTLD_NEXT, "dlsym");

    /* Resolve real CUDA functions */
    real_cuMemAlloc_v2   = (cuMemAlloc_v2_fn)__libc_dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    real_cuMemFree_v2    = (cuMemFree_v2_fn)__libc_dlsym(RTLD_NEXT, "cuMemFree_v2");
    real_cuMemGetInfo_v2 = (cuMemGetInfo_v2_fn)__libc_dlsym(RTLD_NEXT, "cuMemGetInfo_v2");
    real_cuMemAllocAsync = (cuMemAllocAsync_fn)__libc_dlsym(RTLD_NEXT, "cuMemAllocAsync");
    real_cuMemFreeAsync  = (cuMemFreeAsync_fn)__libc_dlsym(RTLD_NEXT, "cuMemFreeAsync");
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

/* --- dlsym interceptor ---
 *
 * CUDA runtime (libcudart.so) resolves driver API symbols via:
 *   handle = dlopen("libcuda.so", ...);
 *   fn = dlsym(handle, "cuMemAlloc_v2");
 *
 * This bypasses LD_PRELOAD because dlsym(handle, ...) searches only that
 * library. We intercept dlsym itself to redirect CUDA symbol lookups
 * to our wrapper functions.
 */
void *dlsym(void *handle, const char *symbol) {
    init();

    /* Redirect CUDA driver API lookups to our wrappers */
    if (strcmp(symbol, "cuMemAlloc_v2") == 0)    return (void *)cuMemAlloc_v2;
    if (strcmp(symbol, "cuMemAlloc") == 0)        return (void *)cuMemAlloc;
    if (strcmp(symbol, "cuMemFree_v2") == 0)      return (void *)cuMemFree_v2;
    if (strcmp(symbol, "cuMemFree") == 0)          return (void *)cuMemFree;
    if (strcmp(symbol, "cuMemGetInfo_v2") == 0)    return (void *)cuMemGetInfo_v2;
    if (strcmp(symbol, "cuMemGetInfo") == 0)        return (void *)cuMemGetInfo;
    if (strcmp(symbol, "cuMemAllocAsync") == 0)    return (void *)cuMemAllocAsync;
    if (strcmp(symbol, "cuMemFreeAsync") == 0)      return (void *)cuMemFreeAsync;

    /* Everything else: pass through to real dlsym */
    if (real_dlsym)
        return real_dlsym(handle, symbol);
    return __libc_dlsym(handle, symbol);
}

/* --- Intercepted CUDA functions: sync --- */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    init();
    if (!real_cuMemAlloc_v2) return CUDA_ERROR_OUT_OF_MEMORY;

    pthread_mutex_lock(&g_lock);

    if (g_limit_bytes > 0 && g_used_bytes + bytesize > g_limit_bytes) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult res = real_cuMemAlloc_v2(dptr, bytesize);
    if (res == CUDA_SUCCESS) {
        g_used_bytes += bytesize;
        track_alloc(*dptr, bytesize);
    }

    pthread_mutex_unlock(&g_lock);
    return res;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    init();
    if (!real_cuMemFree_v2) return CUDA_SUCCESS;

    CUresult res = real_cuMemFree_v2(dptr);

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

/* --- Intercepted CUDA functions: async --- */

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
