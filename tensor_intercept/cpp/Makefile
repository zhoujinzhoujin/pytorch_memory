# Makefile for allocating memory on GPUs

# Set the compiler to use
ifeq ($(CXX),nvcc)
	CC = g++
else
	CC = hipcc
endif


# Set the compiler flags
CFLAGS = -fPIC
CUDA_CFLAGS = -I/usr/local/cuda/include
HIP_CFLAGS = -isystem /opt/rocm-5.4.2/include

# Set the linker flags
LDFLAGS =
CUDA_LDFLAGS = -lcudart
#HIP_LDFLAGS = -L/opt/rocm-5.4.2/lib -lhip_runtime 

# Set the source file
SRC = alloc.cpp

# Set the output file name
OUT = alloc.so

# Define the CUDA and HIP flags based on the __CUDACC__ macro
ifeq ($(CXX),nvcc)
	CFLAGS += $(CUDA_CFLAGS)
	LDFLAGS += $(CUDA_LDFLAGS)
else
	CFLAGS += $(HIP_CFLAGS)
	LDFLAGS += $(HIP_LDFLAGS)
endif

# Define the build target
all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS) -shared

# Define the clean target
clean:
	rm -f $(OUT)

