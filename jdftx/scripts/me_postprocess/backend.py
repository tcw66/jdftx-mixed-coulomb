"""Optional CuPy CUDA backend; files and public result dictionaries use NumPy."""
import numpy as np


def array_module(array):
    if hasattr(array, '__cuda_array_interface__'):
        import cupy
        return cupy
    return np


def host(array):
    return array.get() if hasattr(array, '__cuda_array_interface__') else np.asarray(array)


def select_backend(name='cpu', device=0):
    return name if isinstance(name, Backend) else Backend(name, device)


class Backend:
    def __init__(self, name='cpu', device=0):
        if name not in ('cpu', 'cuda', 'auto'):
            raise ValueError('backend must be cpu, cuda or auto')
        if not isinstance(device, (int, np.integer)) or device < 0:
            raise ValueError('gpu_device must be a nonnegative integer')
        self.name, self.xp = 'cpu', np
        self.info = dict(requested=name, backend='cpu')
        if name == 'cpu':
            return
        try:
            import cupy as cp
            cp.cuda.Device(device).use()
            # Exercise allocation and compilation, not just device enumeration.
            cp.zeros(1, dtype=cp.float64).sum().item()
            props = cp.cuda.runtime.getDeviceProperties(device)
        except Exception as error:
            message = f'CUDA backend unavailable: {error}. Install a matching CuPy/CUDA build or use --backend cpu.'
            if name == 'cuda':
                raise ValueError(message) from error
            self.info['fallback_reason'] = message
            return
        self.name, self.xp = 'cuda', cp
        label = props['name']
        self.info.update(backend='cuda', gpu_device=int(device), cupy_version=cp.__version__,
                         device_name=label.decode() if isinstance(label, bytes) else label)

    def synchronize(self):
        if self.name == 'cuda':
            self.xp.cuda.get_current_stream().synchronize()
