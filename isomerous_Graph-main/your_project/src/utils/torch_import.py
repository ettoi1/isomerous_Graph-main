"""Utility layer that exposes PyTorch when available and a tiny stub otherwise.

The project targets real PyTorch for any meaningful experiment.  When the actual
``torch`` package is missing (which happens inside the execution sandbox used by
these smoke tests) we fall back to a deliberately small shim that mimics the
symbols our bootstrap code imports.  The shim is intentionally lightweight: it
provides tensor placeholders, no-op neural network modules, and stub optimisers
so the application can bootstrap, print user friendly warnings, and exit
gracefully.  All numerics in the stub are dummy values; developers must install
real PyTorch (and numpy) to obtain useful model behaviour.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import random
from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sequence


if TYPE_CHECKING:  # pragma: no cover - import resolved only for type checkers.
    from torch.utils.data import DataLoader as TorchDataLoader  # noqa: F401
    from torch.utils.data import Dataset as TorchDataset  # noqa: F401


try:  # pragma: no cover - exercised only when the genuine dependency exists.
    import torch as _torch  # type: ignore
    import torch.nn as _nn  # type: ignore
    import torch.nn.functional as _F  # type: ignore
    import torch.optim as _optim  # type: ignore
    from torch.utils.data import DataLoader as _DataLoader  # type: ignore
    from torch.utils.data import Dataset as _Dataset  # type: ignore

    TORCH_AVAILABLE = True

    torch = _torch
    nn = _nn
    F = _F
    optim = _optim
    DataLoader = _DataLoader
    Dataset = _Dataset
except ImportError:  # pragma: no cover - this branch is exercised in CI here.
    TORCH_AVAILABLE = False

    # ------------------------------------------------------------------
    # Minimal tensor placeholder ---------------------------------------
    # ------------------------------------------------------------------
    @dataclass
    class _SimpleTensor:
        data: Any

        @property
        def shape(self):
            if isinstance(self.data, (list, tuple)):
                if len(self.data) == 0:
                    return (0,)
                if isinstance(self.data[0], (list, tuple)):
                    inner = _SimpleTensor(self.data[0]).shape
                    return (len(self.data),) + inner
                return (len(self.data),)
            return ()

        @property
        def device(self) -> str:
            return "cpu"

        def to(self, dtype: Optional[Any] = None, device: Optional[str] = None):
            return self

        def float(self):
            return self

        def long(self):
            if isinstance(self.data, (int, float)):
                return _SimpleTensor(int(self.data))
            if isinstance(self.data, list):
                return _SimpleTensor([int(x) for x in self.data])
            return self

        def unsqueeze(self, dim: int):
            if dim == 0:
                return _SimpleTensor([self.data])
            return self

        def squeeze(self, dim: Optional[int] = None):
            if dim in (0, None) and isinstance(self.data, list) and len(self.data) == 1:
                return _SimpleTensor(self.data[0])
            return self

        def sum(self, dim: Optional[int] = None, keepdim: bool = False):
            if isinstance(self.data, list):
                total = sum(float(x) for x in _flatten(self.data))
            else:
                total = float(self.data)
            if keepdim:
                return _SimpleTensor([[total]])
            return _SimpleTensor(total)

        def mean(self, dim: Optional[int] = None, keepdim: bool = False):
            flattened = _flatten(self.data)
            if flattened:
                value = sum(flattened) / len(flattened)
            else:
                value = 0.0
            if keepdim:
                return _SimpleTensor([[value]])
            return _SimpleTensor(value)

        def std(self, dim: Optional[int] = None, keepdim: bool = False):
            flattened = _flatten(self.data)
            if not flattened:
                value = 0.0
            else:
                mean_val = sum(flattened) / len(flattened)
                value = (sum((x - mean_val) ** 2 for x in flattened) / max(len(flattened) - 1, 1)) ** 0.5
            if keepdim:
                return _SimpleTensor([[value]])
            return _SimpleTensor(value)

        def clamp(self, min: Optional[float] = None, max: Optional[float] = None):
            return _SimpleTensor([_clamp_scalar(x, min, max) for x in _flatten(self.data)])

        def view(self, *shape: int):
            return self

        def reshape(self, *shape: int):
            return self

        def transpose(self, *dims: int):
            if isinstance(self.data, list):
                transposed = list(zip(*self.data))
                return _SimpleTensor([list(row) for row in transposed])
            return self

        def t(self):
            return self.transpose()

        def size(self, dim: Optional[int] = None):
            shp = self.shape
            if dim is None:
                return shp
            return shp[dim]

        def item(self):
            if isinstance(self.data, (int, float)):
                return self.data
            if isinstance(self.data, list) and self.data:
                return _SimpleTensor(self.data[0]).item()
            return 0.0

        def detach(self):
            return _SimpleTensor(self.data)

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def index_add(self, dim: int, index: "_SimpleTensor", src: "_SimpleTensor"):
            return self

        def abs(self):
            if isinstance(self.data, list):
                return _SimpleTensor([abs(x) for x in _flatten(self.data)])
            return _SimpleTensor(abs(self.data))

        def any(self):
            return any(bool(x) for x in _flatten(self.data))

        def __getitem__(self, item):
            if isinstance(self.data, list):
                return _SimpleTensor(self.data[item])
            return _SimpleTensor(self.data)

        def __len__(self):
            if isinstance(self.data, list):
                return len(self.data)
            return 1

        def __iter__(self) -> Iterator[Any]:
            if isinstance(self.data, list):
                for val in self.data:
                    yield _SimpleTensor(val)
            else:
                yield self

        def __add__(self, other: Any):
            return _SimpleTensor(self.item() + _to_scalar(other))

        __radd__ = __add__

        def __sub__(self, other: Any):
            return _SimpleTensor(self.item() - _to_scalar(other))

        def __rsub__(self, other: Any):
            return _SimpleTensor(_to_scalar(other) - self.item())

        def __mul__(self, other: Any):
            return _SimpleTensor(self.item() * _to_scalar(other))

        __rmul__ = __mul__

        def __truediv__(self, other: Any):
            denom = _to_scalar(other)
            if denom == 0:
                return _SimpleTensor(0.0)
            return _SimpleTensor(self.item() / denom)

        def __matmul__(self, other: Any):  # pragma: no cover - rarely triggered
            return _SimpleTensor(self.item())

        def backward(self):  # pragma: no cover - gradients unsupported
            return None

        def clone(self):
            return _SimpleTensor(self.data)

        def __repr__(self):  # pragma: no cover - debug helper
            return f"_SimpleTensor({self.data!r})"


    def _flatten(data: Any) -> List[float]:
        if isinstance(data, (list, tuple)):
            values: List[float] = []
            for item in data:
                values.extend(_flatten(item))
            return values
        return [float(data)]


    def _to_scalar(value: Any) -> float:
        if isinstance(value, _SimpleTensor):
            return float(value.item())
        return float(value)


    def _clamp_scalar(value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value


    def tensor(data: Any, dtype: Optional[Any] = None, device: Optional[str] = None):
        return _SimpleTensor(data)


    def from_numpy(array: Any):
        return _SimpleTensor(array)


    def FloatTensor(data: Any):
        return _SimpleTensor(data)


    def LongTensor(data: Any):
        if isinstance(data, list):
            return _SimpleTensor([int(x) for x in data])
        return _SimpleTensor(int(data))


    def zeros(shape: Sequence[int], dtype: Optional[Any] = None, device: Optional[str] = None):
        if len(shape) == 2:
            return _SimpleTensor([[0.0 for _ in range(shape[1])] for _ in range(shape[0])])
        return _SimpleTensor([0.0 for _ in range(shape[0])])


    def zeros_like(other: _SimpleTensor, dtype: Optional[Any] = None):
        if isinstance(other.data, list):
            return _SimpleTensor([[0.0 for _ in row] for row in other.data])
        return _SimpleTensor(0.0)


    def rand(shape: Sequence[int], dtype: Optional[Any] = None, device: Optional[str] = None):
        if len(shape) == 2:
            return _SimpleTensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])])
        return _SimpleTensor([random.random() for _ in range(shape[0])])


    def randn(shape: Sequence[int], dtype: Optional[Any] = None, device: Optional[str] = None):
        if len(shape) == 2:
            return _SimpleTensor([[random.gauss(0.0, 1.0) for _ in range(shape[1])] for _ in range(shape[0])])
        return _SimpleTensor([random.gauss(0.0, 1.0) for _ in range(shape[0])])


    def randint(low: int, high: int, size: Sequence[int], dtype: Optional[Any] = None, device: Optional[str] = None):
        if len(size) == 2:
            return _SimpleTensor([[random.randint(low, high - 1) for _ in range(size[1])] for _ in range(size[0])])
        return _SimpleTensor([random.randint(low, high - 1) for _ in range(size[0])])


    def arange(*args: Any, dtype: Optional[Any] = None, device: Optional[str] = None):
        if len(args) == 1:
            end = int(args[0])
            values = list(range(end))
        else:
            start, end = int(args[0]), int(args[1])
            step = int(args[2]) if len(args) > 2 else 1
            values = list(range(start, end, step))
        return _SimpleTensor(values)


    def linspace(start: float, end: float, steps: int, dtype: Optional[Any] = None, device: Optional[str] = None):
        if steps <= 1:
            return _SimpleTensor([start])
        interval = (end - start) / (steps - 1)
        return _SimpleTensor([start + i * interval for i in range(steps)])


    def stack(tensors: Sequence[_SimpleTensor], dim: int = 0):
        data = [t.data if isinstance(t, _SimpleTensor) else t for t in tensors]
        return _SimpleTensor(data)


    def cat(tensors: Sequence[_SimpleTensor], dim: int = 0):
        data = []
        for t in tensors:
            data.extend(t.data if isinstance(t, _SimpleTensor) else t)
        return _SimpleTensor(data)


    def topk(input_tensor: _SimpleTensor, k: int, dim: int = -1):
        values = []
        indices = []
        for row_idx, row in enumerate(input_tensor.data):
            pairs = sorted([(val, idx) for idx, val in enumerate(row)], reverse=True)
            chosen = pairs[:k]
            values.append([val for val, _ in chosen])
            indices.append([idx for _, idx in chosen])
        return _SimpleTensor(values), _SimpleTensor(indices)


    def manual_seed(seed: int):
        random.seed(seed)


    class _CudaModule:
        @staticmethod
        def manual_seed_all(seed: int):  # pragma: no cover - shim only
            random.seed(seed)

    cuda = _CudaModule()


    def no_grad():  # pragma: no cover - shim only
        return contextlib.nullcontext()


    def save(obj: Any, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


    def load(path: str):
        with open(path, "rb") as fh:
            return pickle.load(fh)


    def argmax(tensor: _SimpleTensor, dim: int = 0):
        if isinstance(tensor.data, list):
            max_idx = max(range(len(tensor.data)), key=lambda idx: tensor.data[idx])
            return _SimpleTensor(max_idx)
        return _SimpleTensor(0)


    def bincount(arr: _SimpleTensor, minlength: int = 0):
        counts: List[int] = [0 for _ in range(minlength)]
        for value in _flatten(arr.data):
            idx = int(value)
            while idx >= len(counts):
                counts.append(0)
            counts[idx] += 1
        return _SimpleTensor(counts)


    float32 = float
    long = int
    int64 = int
    bool = bool

    class Module:
        def __init__(self):
            self.training = True

        def train(self, mode: bool = True):
            self.training = mode
            return self

        def eval(self):
            return self.train(False)

        def parameters(self) -> List[_SimpleTensor]:
            return []

        def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - abstract
            raise NotImplementedError

        def __call__(self, *args: Any, **kwargs: Any):
            return self.forward(*args, **kwargs)


    class Linear(Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x: _SimpleTensor) -> _SimpleTensor:
            return x


    class ReLU(Module):
        def forward(self, x: _SimpleTensor) -> _SimpleTensor:
            return x


    class Sequential(Module):
        def __init__(self, *modules: Module):
            super().__init__()
            self.modules = modules

        def forward(self, x: _SimpleTensor) -> _SimpleTensor:
            out = x
            for module in self.modules:
                out = module(out)
            return out


    class ModuleList(Module):
        def __init__(self, modules: Iterable[Module]):
            super().__init__()
            self._modules = list(modules)

        def __iter__(self) -> Iterator[Module]:
            return iter(self._modules)


    class CrossEntropyLoss(Module):
        def forward(self, input_tensor: _SimpleTensor, target: _SimpleTensor) -> _SimpleTensor:
            return _SimpleTensor(0.0)


    class Identity(Module):
        def forward(self, x: _SimpleTensor) -> _SimpleTensor:
            return x


    def _softmax(x: _SimpleTensor, dim: int = -1) -> _SimpleTensor:
        return x


    def _relu(x: _SimpleTensor) -> _SimpleTensor:
        return x


    class _Adam:
        def __init__(self, params: Iterable[_SimpleTensor], lr: float = 1e-3):
            self.params = list(params)
            self.lr = lr

        def zero_grad(self):
            return None

        def step(self):
            return None


    class _Dataset:
        def __len__(self) -> int:  # pragma: no cover - abstract
            raise NotImplementedError

        def __getitem__(self, idx: int):  # pragma: no cover - abstract
            raise NotImplementedError


    class _DataLoader:
        def __init__(
            self,
            dataset: _Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            collate_fn: Optional[Any] = None,
        ):
            self.dataset = dataset
            self.batch_size = max(1, batch_size)
            self.shuffle = shuffle
            self.collate_fn = collate_fn

        def __iter__(self) -> Iterator[Any]:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for idx in indices:
                sample = self.dataset[idx]
                if self.collate_fn is None:
                    yield sample
                else:
                    yield self.collate_fn([sample])

        def __len__(self) -> int:
            return len(self.dataset)


    torch = type(
        "torch",
        (),
        {
            "Tensor": _SimpleTensor,
            "tensor": staticmethod(tensor),
            "from_numpy": staticmethod(from_numpy),
            "FloatTensor": staticmethod(FloatTensor),
            "LongTensor": staticmethod(LongTensor),
            "zeros": staticmethod(zeros),
            "zeros_like": staticmethod(zeros_like),
            "rand": staticmethod(rand),
            "randn": staticmethod(randn),
            "randint": staticmethod(randint),
            "arange": staticmethod(arange),
            "linspace": staticmethod(linspace),
            "stack": staticmethod(stack),
            "cat": staticmethod(cat),
            "topk": staticmethod(topk),
            "manual_seed": staticmethod(manual_seed),
            "cuda": cuda,
            "no_grad": staticmethod(no_grad),
            "save": staticmethod(save),
            "load": staticmethod(load),
            "argmax": staticmethod(argmax),
            "bincount": staticmethod(bincount),
            "float32": float32,
            "long": long,
            "int64": int64,
            "bool": bool,
        },
    )()

    nn = type(
        "nn",
        (),
        {
            "Module": Module,
            "Linear": Linear,
            "ReLU": ReLU,
            "Sequential": Sequential,
            "ModuleList": ModuleList,
            "CrossEntropyLoss": CrossEntropyLoss,
            "Identity": Identity,
        },
    )()

    F = type(
        "functional",
        (),
        {"softmax": staticmethod(_softmax), "relu": staticmethod(_relu)},
    )()

    optim = type("optim", (), {"Adam": _Adam})()

    Dataset = _Dataset
    DataLoader = _DataLoader


__all__ = ["TORCH_AVAILABLE", "torch", "nn", "F", "optim", "Dataset", "DataLoader"]
