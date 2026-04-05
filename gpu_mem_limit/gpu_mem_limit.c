/*
 * gpu_mem_limit.c — LD_PRELOAD library to enforce per-process GPU memory limits.
 *
 * Intercepts CUDA runtime API AND driver API calls to cap each process
 * at GPU_MEM_LIMIT_MB megabytes of GPU memory.
 *
 * Runtime API: cudaMalloc, cudaFree, cudaMallocAsync, cudaFreeAsync, cudaMemGetInfo
 * Driver API:  cuMemCreate, cuMemRelease, cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2
 *
 * Why both layers?
 *   PyTorch 2.x + CUDA 12.x defaults to "expandable segments" which allocates
 *   physical GPU memory via cuMemCreate (driver API) instead of cudaMalloc.
 *   Intercepting only runtime API misses these allocations entirely.
 *
 * Usage:
 *   GPU_MEM_LIMIT_MB=88000 LD_PRELOAD=./libgpumemlimit.so python train.py
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── CUDA types (defined locally to avoid requiring CUDA headers) ───── */

typedef int cudaError_t;
#define cudaSuccess              0
#define cudaErrorMemoryAllocation 2
typedef void *cudaStream_t;

typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUresult;
#define CUDA_SUCCESS             0
#define CUDA_ERROR_OUT_OF_MEMORY 2

/* ── Function pointer types ──────────────────────────────────────────── */

typedef cudaError_t (*cudaMalloc_fn)(void **, size_t);
typedef cudaError_t (*cudaFree_fn)(void *);
typedef cudaError_t (*cudaMemGetInfo_fn)(size_t *, size_t *);
typedef cudaError_t (*cudaMallocAsync_fn)(void **, size_t, cudaStream_t);
typedef cudaError_t (*cudaFreeAsync_fn)(void *, cudaStream_t);

/* cuMemCreate: prop is opaque const void* — we just pass it through */
typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle *, size_t,
                                   const void *, unsigned long long);
typedef CUresult (*cuMemRelease_fn)(CUmemGenericAllocationHandle);
typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr);
typedef CUresult (*cuMemGetInfo_v2_fn)(size_t *, size_t *);

/* ── Open-addressing hash table (uint64 key → size_t size) ──────────── */

#define HT_INIT_CAP  4096
#define HT_LOAD_MAX  0.7
#define HT_EMPTY     0ULL
#define HT_TOMBSTONE 1ULL

typedef struct { uint64_t key; size_t size; } ht_entry_t;
typedef struct { ht_entry_t *entries; int capacity; int count; } ht_t;

static int ht_probe(ht_t *ht, uint64_t key) {
    unsigned mask = (unsigned)(ht->capacity - 1);
    unsigned idx  = (unsigned)((key * 0x9E3779B97F4A7C15ULL) >> 48) & mask;
    int first_tomb = -1;
    for (int i = 0; i < ht->capacity; i++) {
        unsigned slot = (idx + (unsigned)i) & mask;
        uint64_t k = ht->entries[slot].key;
        if (k == HT_EMPTY)
            return (first_tomb >= 0) ? first_tomb : (int)slot;
        if (k == HT_TOMBSTONE && first_tomb < 0)
            first_tomb = (int)slot;
        else if (k == key)
            return (int)slot;
    }
    return (first_tomb >= 0) ? first_tomb : -1;
}

static void ht_grow(ht_t *ht) {
    int old_cap = ht->capacity;
    ht_entry_t *old = ht->entries;
    ht->capacity *= 2;
    ht->entries = (ht_entry_t *)calloc((size_t)ht->capacity, sizeof(ht_entry_t));
    if (!ht->entries) {
        fprintf(stderr, "[gpu_mem_limit] FATAL: calloc failed during rehash\n");
        abort();
    }
    ht->count = 0;
    for (int i = 0; i < old_cap; i++) {
        if (old[i].key > HT_TOMBSTONE) {
            int s = ht_probe(ht, old[i].key);
            ht->entries[s] = old[i];
            ht->count++;
        }
    }
    free(old);
}

static void ht_insert(ht_t *ht, uint64_t key, size_t size) {
    if (ht->entries == NULL) {
        ht->capacity = HT_INIT_CAP;
        ht->entries = (ht_entry_t *)calloc(HT_INIT_CAP, sizeof(ht_entry_t));
        if (!ht->entries) {
            fprintf(stderr, "[gpu_mem_limit] FATAL: calloc failed during init\n");
            abort();
        }
    }
    if ((double)ht->count / ht->capacity > HT_LOAD_MAX)
        ht_grow(ht);
    int s = ht_probe(ht, key);
    if (s >= 0) {
        if (ht->entries[s].key != key)
            ht->count++;
        ht->entries[s].key  = key;
        ht->entries[s].size = size;
    }
}

static size_t ht_remove(ht_t *ht, uint64_t key) {
    if (ht->entries == NULL) return 0;
    int s = ht_probe(ht, key);
    if (s >= 0 && ht->entries[s].key == key) {
        size_t sz = ht->entries[s].size;
        ht->entries[s].key  = HT_TOMBSTONE;
        ht->entries[s].size = 0;
        ht->count--;
        return sz;
    }
    return 0;
}

/* ── Global state ────────────────────────────────────────────────────── */

static size_t          g_limit_bytes = 0;
static size_t          g_used_bytes  = 0;
static ht_t            g_ht          = {NULL, 0, 0};
static pthread_mutex_t g_lock        = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_once        = PTHREAD_ONCE_INIT;

/* Recursion guard: prevents double-counting when a runtime API call
   (e.g. cudaMalloc) internally dispatches to a driver API call
   (e.g. cuMemAlloc_v2) through the PLT. */
static __thread int g_in_hook = 0;

/* Real function pointers */
static cudaMalloc_fn      real_cudaMalloc      = NULL;
static cudaFree_fn        real_cudaFree        = NULL;
static cudaMemGetInfo_fn  real_cudaMemGetInfo  = NULL;
static cudaMallocAsync_fn real_cudaMallocAsync = NULL;
static cudaFreeAsync_fn   real_cudaFreeAsync   = NULL;
static cuMemCreate_fn     real_cuMemCreate     = NULL;
static cuMemRelease_fn    real_cuMemRelease    = NULL;
static cuMemAlloc_v2_fn   real_cuMemAlloc_v2   = NULL;
static cuMemFree_v2_fn    real_cuMemFree_v2    = NULL;
static cuMemGetInfo_v2_fn real_cuMemGetInfo_v2 = NULL;

/* ── Initialization ──────────────────────────────────────────────────── */

static void do_init(void) {
    const char *env = getenv("GPU_MEM_LIMIT_MB");
    if (env && atol(env) > 0) {
        g_limit_bytes = (size_t)atol(env) * 1024ULL * 1024ULL;
        fprintf(stderr, "[gpu_mem_limit] limit=%s MB (%zu bytes)\n",
                env, g_limit_bytes);
    }

    /* Runtime API */
    real_cudaMalloc      = (cudaMalloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaFree        = (cudaFree_fn)dlsym(RTLD_NEXT, "cudaFree");
    real_cudaMemGetInfo  = (cudaMemGetInfo_fn)dlsym(RTLD_NEXT, "cudaMemGetInfo");
    real_cudaMallocAsync = (cudaMallocAsync_fn)dlsym(RTLD_NEXT, "cudaMallocAsync");
    real_cudaFreeAsync   = (cudaFreeAsync_fn)dlsym(RTLD_NEXT, "cudaFreeAsync");

    /* Driver API — libcuda.so is typically dlopen'd by the runtime,
       so RTLD_NEXT won't find it.  Open it directly (RTLD_NOLOAD
       just grabs the already-loaded handle, no new load). */
    void *cuda_drv = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!cuda_drv)
        cuda_drv = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
    if (cuda_drv) {
        real_cuMemCreate     = (cuMemCreate_fn)dlsym(cuda_drv, "cuMemCreate");
        real_cuMemRelease    = (cuMemRelease_fn)dlsym(cuda_drv, "cuMemRelease");
        real_cuMemAlloc_v2   = (cuMemAlloc_v2_fn)dlsym(cuda_drv, "cuMemAlloc_v2");
        real_cuMemFree_v2    = (cuMemFree_v2_fn)dlsym(cuda_drv, "cuMemFree_v2");
        real_cuMemGetInfo_v2 = (cuMemGetInfo_v2_fn)dlsym(cuda_drv, "cuMemGetInfo_v2");
        dlclose(cuda_drv);
    }

    /* Log which functions we found */
    fprintf(stderr, "[gpu_mem_limit] intercepting:");
    if (real_cudaMalloc)      fprintf(stderr, " cudaMalloc");
    if (real_cudaFree)        fprintf(stderr, " cudaFree");
    if (real_cudaMemGetInfo)  fprintf(stderr, " cudaMemGetInfo");
    if (real_cudaMallocAsync) fprintf(stderr, " cudaMallocAsync");
    if (real_cudaFreeAsync)   fprintf(stderr, " cudaFreeAsync");
    if (real_cuMemCreate)     fprintf(stderr, " cuMemCreate");
    if (real_cuMemRelease)    fprintf(stderr, " cuMemRelease");
    if (real_cuMemAlloc_v2)   fprintf(stderr, " cuMemAlloc_v2");
    if (real_cuMemFree_v2)    fprintf(stderr, " cuMemFree_v2");
    if (real_cuMemGetInfo_v2) fprintf(stderr, " cuMemGetInfo_v2");
    fprintf(stderr, "\n");
}

static void init(void) { pthread_once(&g_once, do_init); }

/* ── Helpers ─────────────────────────────────────────────────────────── */

/* Check limit and account for new allocation. Returns 1 if allowed, 0 if denied.
   Caller MUST hold g_lock. */
static int try_alloc(size_t size) {
    if (g_limit_bytes > 0 && g_used_bytes + size > g_limit_bytes)
        return 0;
    g_used_bytes += size;
    return 1;
}

/* Account for a freed allocation. Caller MUST hold g_lock. */
static void do_free(uint64_t key) {
    size_t freed = ht_remove(&g_ht, key);
    if (freed > g_used_bytes)
        g_used_bytes = 0;
    else
        g_used_bytes -= freed;
}

/* ── Runtime API interception ────────────────────────────────────────── */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    init();
    if (!real_cudaMalloc) return cudaErrorMemoryAllocation;

    pthread_mutex_lock(&g_lock);
    if (!try_alloc(size)) {
        pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }
    g_in_hook = 1;
    cudaError_t res = real_cudaMalloc(devPtr, size);
    g_in_hook = 0;
    if (res == cudaSuccess) {
        ht_insert(&g_ht, (uint64_t)(uintptr_t)*devPtr, size);
    } else {
        g_used_bytes -= size;  /* rollback */
    }
    pthread_mutex_unlock(&g_lock);
    return res;
}

cudaError_t cudaFree(void *devPtr) {
    init();
    if (!real_cudaFree) return cudaSuccess;

    g_in_hook = 1;
    cudaError_t res = real_cudaFree(devPtr);
    g_in_hook = 0;
    if (res == cudaSuccess && devPtr) {
        pthread_mutex_lock(&g_lock);
        do_free((uint64_t)(uintptr_t)devPtr);
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}

cudaError_t cudaMemGetInfo(size_t *free_out, size_t *total_out) {
    init();
    if (!real_cudaMemGetInfo) return 1;

    g_in_hook = 1;
    cudaError_t res = real_cudaMemGetInfo(free_out, total_out);
    g_in_hook = 0;
    if (res == cudaSuccess && g_limit_bytes > 0) {
        pthread_mutex_lock(&g_lock);
        *total_out = g_limit_bytes;
        *free_out  = (g_used_bytes < g_limit_bytes)
                   ? g_limit_bytes - g_used_bytes : 0;
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    init();
    if (!real_cudaMallocAsync) return cudaErrorMemoryAllocation;

    pthread_mutex_lock(&g_lock);
    if (!try_alloc(size)) {
        pthread_mutex_unlock(&g_lock);
        return cudaErrorMemoryAllocation;
    }
    g_in_hook = 1;
    cudaError_t res = real_cudaMallocAsync(devPtr, size, stream);
    g_in_hook = 0;
    if (res == cudaSuccess) {
        ht_insert(&g_ht, (uint64_t)(uintptr_t)*devPtr, size);
    } else {
        g_used_bytes -= size;
    }
    pthread_mutex_unlock(&g_lock);
    return res;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    init();
    if (!real_cudaFreeAsync) return cudaSuccess;

    g_in_hook = 1;
    cudaError_t res = real_cudaFreeAsync(devPtr, stream);
    g_in_hook = 0;
    if (res == cudaSuccess && devPtr) {
        pthread_mutex_lock(&g_lock);
        do_free((uint64_t)(uintptr_t)devPtr);
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}

/* ── Driver API interception ─────────────────────────────────────────── */

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const void *prop, unsigned long long flags) {
    init();
    if (!real_cuMemCreate) return CUDA_ERROR_OUT_OF_MEMORY;
    if (g_in_hook) return real_cuMemCreate(handle, size, prop, flags);

    pthread_mutex_lock(&g_lock);
    if (!try_alloc(size)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = real_cuMemCreate(handle, size, prop, flags);
    if (res == CUDA_SUCCESS) {
        ht_insert(&g_ht, (uint64_t)*handle, size);
    } else {
        g_used_bytes -= size;
    }
    pthread_mutex_unlock(&g_lock);
    return res;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    init();
    if (!real_cuMemRelease) return CUDA_SUCCESS;
    if (g_in_hook) return real_cuMemRelease(handle);

    CUresult res = real_cuMemRelease(handle);
    if (res == CUDA_SUCCESS) {
        pthread_mutex_lock(&g_lock);
        do_free((uint64_t)handle);
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    init();
    if (!real_cuMemAlloc_v2) return CUDA_ERROR_OUT_OF_MEMORY;
    if (g_in_hook) return real_cuMemAlloc_v2(dptr, size);

    pthread_mutex_lock(&g_lock);
    if (!try_alloc(size)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = real_cuMemAlloc_v2(dptr, size);
    if (res == CUDA_SUCCESS) {
        ht_insert(&g_ht, (uint64_t)*dptr, size);
    } else {
        g_used_bytes -= size;
    }
    pthread_mutex_unlock(&g_lock);
    return res;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    init();
    if (!real_cuMemFree_v2) return CUDA_SUCCESS;
    if (g_in_hook) return real_cuMemFree_v2(dptr);

    CUresult res = real_cuMemFree_v2(dptr);
    if (res == CUDA_SUCCESS && dptr != 0) {
        pthread_mutex_lock(&g_lock);
        do_free((uint64_t)dptr);
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}

CUresult cuMemGetInfo_v2(size_t *free_out, size_t *total_out) {
    init();
    if (!real_cuMemGetInfo_v2) return 1;
    if (g_in_hook) return real_cuMemGetInfo_v2(free_out, total_out);

    CUresult res = real_cuMemGetInfo_v2(free_out, total_out);
    if (res == CUDA_SUCCESS && g_limit_bytes > 0) {
        pthread_mutex_lock(&g_lock);
        *total_out = g_limit_bytes;
        *free_out  = (g_used_bytes < g_limit_bytes)
                   ? g_limit_bytes - g_used_bytes : 0;
        pthread_mutex_unlock(&g_lock);
    }
    return res;
}
