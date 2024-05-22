from ml_logger import logger
from functools import reduce
from inspect import isfunction
from typing import Callable, Optional, Sequence, TypeVar, Union
from typing_extensions import TypeGuard
import torch

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


class TimedAction:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.start(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = logger.split(self.name)
        logger.print(f'Action {self.name} took {t:.3}s')


def reduce_dim_except_0(A, f):
    return f(A, dim=tuple(range(1, len(A.shape))))


def pad_dimensions(A, B):
    """given 1-dim tensor A, we append dimensions of size 1
    so the number of dimensions matches B
    """
    return A.view([-1] + [1] * (len(B.shape) - 1))


def torchmodify(name):
    a = name.split('.')
    for i, s in enumerate(a):
        if s.isnumeric():
            a[i] = "_modules['" + s + "']"
    return '.'.join(a)


def patch_model(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.GELU):
            exec('model.' + torchmodify(name) + '=torch.nn.GELU()')
