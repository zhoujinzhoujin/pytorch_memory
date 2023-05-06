#include <cstdio>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#endif

extern "C" {
void* my_malloc(ssize_t size, int device, void* stream) {
   if(size == 0) {
        return nullptr;
   }
   void *ptr;
#ifdef __CUDACC__
   cudaSetDevice(device);
   cudaMalloc(&ptr, size);
#else
   hipSetDevice(device);
   hipMalloc(&ptr, size);
#endif
   fprintf(stderr, "plug alloc %lu at %p\n", size, ptr);
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, void* stream) {
   fprintf(stderr, "plug free %lu at %p\n", size, ptr);
#ifdef __CUDACC__
   cudaFree(ptr);
#else
   hipFree(ptr);
#endif
}
}

