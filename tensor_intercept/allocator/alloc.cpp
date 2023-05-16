#include <cstdio>
#include <cassert>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#endif
#include <unordered_map>

#define TORCH_CHECK(condition, ...)

struct Info {
    int stage;
    int layer;
    Info() : stage(0), layer(0) {}

    // Equality comparison function
    bool operator==(const Info& other) const {
        return stage == other.stage && layer == other.layer;
    }
};

// Hash function for Info struct
namespace std {
    template <>
    struct hash<Info> {
        std::size_t operator()(const Info& info) const {
            std::size_t h1 = std::hash<int>{}(info.stage);
            std::size_t h2 = std::hash<int>{}(info.layer);
            return h1 ^ (h2 << 1); // Combine hashes
        }
    };
}

struct Object {
   void * ptr;
   size_t sz;
};

class MemoryManager {
public:
    MemoryManager() : initialized(false), memory_pool(nullptr), current_offset(0), ptr_map() {}

void init(int device) {
    assert(device == 0 && "Device must be cuda:0.");
#ifdef __CUDACC__
        cudaSetDevice(device);
        cudaMemGetInfo(&free_memory, &total_memory);
#else
        hipSetDevice(device);
        hipMemGetInfo(&free_memory, &total_memory);
#endif
        adjusted_memory = free_memory - (5ull * 1024 * 1024 * 1024);
#ifdef __CUDACC__
        cudaMalloc(&memory_pool, adjusted_memory);
        cudaMalloc(&info, sizeof(Info));
#else
        hipMalloc(&memory_pool, adjusted_memory);
        hipMalloc(&info, sizeof(Info));
#endif
        assert(memory_pool != nullptr && "Memory pool allocation failed.");
        assert(info != nullptr && "Info allocation failed.");

        // Calculate the end address of the memory pool
        void* end_address = static_cast<void*>(static_cast<char*>(memory_pool) + adjusted_memory);

        // Print the memory pool range and size in GB
        fprintf(stderr, "Memory pool range: %p - %p, size: %.2f GB\n", memory_pool, end_address, adjusted_memory / (1024.0 * 1024 * 1024));

        initialized = true;
}


   void* allocate(ssize_t size) {
      void* ptr;
      update_info_layer();
      if(ptr_map.find(*reinterpret_cast<struct Info*>(info)) == ptr_map.end()) {
         if (current_offset + size > adjusted_memory) {
            assert(false && "Memory allocation request exceeds available memory."); 
            return nullptr;
         }
         ptr = static_cast<void*>(static_cast<char*>(memory_pool) + current_offset);
         ptr_map[*reinterpret_cast<struct Info*>(info)].ptr = ptr;
         ptr_map[*reinterpret_cast<struct Info*>(info)].sz = size;
         current_offset += size;
      } else {
         ptr = ptr_map[*reinterpret_cast<struct Info*>(info)].ptr; 
         if(size > ptr_map[*reinterpret_cast<struct Info*>(info)].sz) {
            fprintf(stderr, "size %lu, new size %lu\n", ptr_map[*reinterpret_cast<struct Info*>(info)].sz, size);
            assert(false && "size >= ptr_map[*reinterpret_cast<struct Info*>(info)].sz");
         }
      }
      return ptr;
   }

    void deallocate(void* ptr, ssize_t size) {
        // Do nothing
    }

    size_t remaining_memory() const {
        return adjusted_memory - current_offset;
    }

   bool is_initialized() const {
        return initialized;
    }

    void* get_info_addr() const {
        return info;
    }

    int get_info_stage() const {
      return reinterpret_cast<struct Info*>(info)->stage;
    }

    int get_info_layer() const {
      return reinterpret_cast<struct Info*>(info)->layer;
    }

    void update_info_layer() {
      reinterpret_cast<struct Info*>(info)->layer++;
    }

private:
    bool initialized;
    void* memory_pool;
    void* info;
    size_t current_offset;
    size_t free_memory;
    size_t total_memory;
    size_t adjusted_memory;
    std::unordered_map<struct Info, struct Object> ptr_map;
};

static MemoryManager memory_manager;

extern "C" {
void* my_malloc(ssize_t size, int device, void* stream) {
   assert(device == 0 && "Device must be cuda:0.");
    if (size == 0) {
        return nullptr;
    }
    if(memory_manager.is_initialized() == false) {
      memory_manager.init(device);
      return memory_manager.get_info_addr();
    }
    void *ptr = memory_manager.allocate(size);
    size_t remaining_memory = memory_manager.remaining_memory();
    fprintf(stderr, "plug alloc %lu at %p, stage %d layer %d, remaining memory: %lu\n", size, ptr, 
            memory_manager.get_info_stage(),memory_manager.get_info_layer(), remaining_memory);
    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, void* stream) {
   assert(device == 0 && "Device must be cuda:0.");
    fprintf(stderr, "plug free %lu at %p\n", size, ptr);
    memory_manager.deallocate(ptr, size);
}
}
