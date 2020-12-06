import torch
import numpy as np
from abc import abstractmethod
from typing import Type
from skimage import io
from torch import Tensor

class MetaData(object):
    def __init__(self, name, tensor, **kwargs):
        self._name = name
        self._tensor = tensor
        self.kwargs = kwargs

    @classmethod
    def make_empty_as(cls, k: 'MetaData'):
        return k.__class__(k._name, None, **k.kwargs)

    @classmethod
    def make_from_tensor(cls, name, tensor: Tensor, **kwargs):
        return cls(name, tensor, **kwargs)

    @abstractmethod
    def _ToTensor(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_item(cls, name, item, **kwargs):
        raise NotImplementedError

    @classmethod
    def _ToBatch(cls, datas, key):
        assert len(datas) > 0 and isinstance(datas, list), type(datas)
        assert key == datas[0][key]._name, "key different: {} != {}".format(key, datas[0][key]._name)

        samples = MetaData.make_empty_as(datas[0][key])

        samples._tensor = torch.stack([data[key]._tensor for data in datas]).contiguous()
        return samples

    @abstractmethod
    def visualize(self):
        raise NotImplementedError

    @abstractmethod
    def write_file(self):
        raise NotImplementedError

    @abstractmethod
    def write_tensorboard(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(self, loss, gt):
        return loss(self._tensor, gt)

    @abstractmethod
    def _ToTensor(self):
        raise NotImplementedError

    @abstractmethod
    def _ToNumpy(self):
        raise NotImplementedError

class ImageMeta(MetaData):
    def __init__(self, name, tensor=None, **kwargs):
        super().__init__(name, tensor, **kwargs)

    @classmethod
    def load_from_item(cls, name, item, **kwargs):
        """:item: "img file path"
        """
        img = io.imread(item)
        return cls(name, img, **kwargs)

    def _device(self, d):
        self._tensor.device(d)

    def _ToTensor(self):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if isinstance(self._tensor, np.ndarray):
            if self._tensor.ndim == 3:
                self._tensor = torch.from_numpy(self._tensor.transpose((2, 0, 1)))
            elif self._tensor.ndim == 4:
                self._tensor = torch.from_numpy(self._tensor.transpose((0, 3, 1, 2)))
            else:
                raise ValueError('Image ndim must to be 3 or 4')
        return self._tensor

    def _ToNumpy(self):
        # swap color axis because
        # torch image: C X H X W
        # numpy image: H x W x C

        if isinstance(self._tensor, Tensor):
            if self._tensor.ndim == 3:
                self._tensor = self._tensor.permute((1, 2, 0)).cpu().numpy()
            elif self._tensor.ndim == 4:
                self._tensor = self._tensor.permute((0, 2, 3, 1)).cpu().numpy()
            else:
                raise ValueError('Image ndim must to be 3 or 4')
        return self._tensor
