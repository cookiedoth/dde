import torch
from ml_logger import logger
import numpy as np


def to_dict(proto):
	return {k: v for k, v in vars(proto).items() if not callable(v) and not k.startswith('__')}


class TimedAction:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.start(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = logger.split(self.name)
        logger.print(f'Action {self.name} took {t:.3}s')


def pad_dimensions(A, B):
    """given 1-dim tensor A, we append dimensions of size 1 
    so the number of dimensions matches B
    """
    return A.view([-1] + [1] * (len(B.shape) - 1))


def reduce_dim_except_0(A, f):
    return f(A, dim=tuple(range(1, len(A.shape))))


def tensor_to_image(tensor):
    """tensor is of shape [3, H, W], in [-1, 1] range
    returns numpy suitable for plt.imshow
    """
    np_image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    np_image = (np_image + 1) / 2.0
    return np_image


def torch_randint(l, r):
    return torch.randint(low=l, high=r, size=(1,)).item()


def to_tuple2(x):
    if isinstance(x, tuple) and len(x) == 2:
        return x
    elif isinstance(x, int):
        return (x, x)
    else:
        raise ValueError(f'{x} should be int or a 2-tuple')


def np_stats(arr, name=None):
    logger.print(f'Array{"" if name is None else " " + name}; ',
                 'shape: ', arr.shape, ' ',
                 'dtype: ', str(arr.dtype), ' ',
                 'range: ', np.min(arr), '..', np.max(arr), ' ',
                 'mean: ', np.mean(arr), ' ',
                 'std: ', np.std(arr), sep='')


def torch_stats(arr, name=None):
    logger.print(f'Array{"" if name is None else " " + name}; ',
                 'shape: ', arr.shape, ' ',
                 'dtype: ', str(arr.dtype), ' ',
                 'range: ', torch.min(arr), '..', torch.max(arr), ' ',
                 'mean: ', torch.mean(arr), ' ',
                 'std: ', torch.std(arr), sep='')
