import torch
import torchvision
from typing import Any, Iterable, List, Tuple
import checkpoint2

torch.utils.checkpoint.checkpoint = checkpoint2.checkpoint_2

        
