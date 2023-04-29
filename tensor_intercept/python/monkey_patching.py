'''

python-level interception for tensor construction

'''


import torch
import inspect
import pkgutil
import importlib
import torchvision
import sys


def wrap_with_logging(func):


    def logging_wrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        if isinstance(result, torch.Tensor) and result.is_cuda:
            hex_address = format(result.data_ptr(), 'x')
            tensor_size_bytes = result.element_size() * result.numel()
            print(f'{func.__name__}, shape {result.size()}, size {tensor_size_bytes}, address 0x{hex_address}', file=sys.stderr)

        return result


    return logging_wrapper


'''
intercept tensor constructions in model and data initialization
'''

torch.empty = wrap_with_logging(torch.empty)
torch.Tensor.to = wrap_with_logging(torch.Tensor.to)


'''
intercept tensor constrcutions in model's forward(), and loss computation (miss some constructions)
'''

for _, module_name, _ in pkgutil.walk_packages(torch.nn.__path__):
    importlib.import_module(f"{torch.nn.__name__}.{module_name}")

layer_classes = [cls for name, cls in inspect.getmembers(torch.nn, inspect.isclass) if issubclass(cls, torch.nn.Module) and cls != torch.nn.Module]

for layer_class in layer_classes:
    methods = inspect.getmembers(layer_class, predicate=inspect.isfunction)
    for method_name, method in methods:
        if not method_name.startswith("_"):
            method_signature = inspect.signature(method)
            return_annotation = method_signature.return_annotation
            if return_annotation == torch.Tensor:
                wrapped_method = wrap_with_logging(method)
                setattr(layer_class, method_name, wrapped_method)


'''
intercept tensor constructions in model's backward() (miss some constructions)
'''

torch.ones_like = wrap_with_logging(torch.ones_like)


'''
intercept tensor constructions in optimizer's step()
'''

torch.Tensor.detach = wrap_with_logging(torch.Tensor.detach)
