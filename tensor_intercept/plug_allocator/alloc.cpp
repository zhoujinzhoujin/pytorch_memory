#include <stdio.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>

extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   if(size == 0) {
	return nullptr;
   }
   void *ptr;
   cudaMalloc(&ptr, size);
   fprintf(stderr, "plug alloc %lu at %p\n", size, ptr);
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   fprintf(stderr, "plug free %lu at %p\n", size, ptr);
   cudaFree(ptr);
}
}
