import os
import torch

# Get the directory containing the current file
file_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the alloc.so file
alloc_path = os.path.join(file_dir, 'alloc.so')

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    alloc_path, 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator

