import torch
import framework.dataloader.transform_detail
from framework.dataloader.MetaData import Data

class BaseTransform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        pass

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, samples):
        for data in samples:
            data._ToTensor()

        return samples

