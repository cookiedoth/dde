import io
import pickle
import torch


class TorchCpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_model_local(filename, device):
    file = open(filename, 'rb')
    if device == 'cpu':
        return TorchCpuUnpickler(file).load()
    elif device == 'cuda':
        return pickle.load(file)
    else:
        raise ValueError('Unknown device')


def load_model(run_path, checkpoint_name):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)
    model = loader.torch_load(checkpoint_name, map_location='cpu')
    return model
