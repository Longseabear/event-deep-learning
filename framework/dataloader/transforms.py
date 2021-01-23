import numpy
from framework.dataloader.TensorTypes import *
class Transforms(object):
    def __init__(self, required=[], meta={}):
        self.required = required
        self.meta = meta
        if 'ALL' in self.meta.keys():
            for name in self.required:
                self.meta[name] = self.meta['ALL']

class ToTensor(Transforms):
    def __init__(self, required=[], meta={}):
        super(ToTensor, self).__init__(required, meta)

    def __call__(self, samples):
        for sample in self.required:
            data = samples[sample]
            meta_data = self.meta[sample]

            if isinstance(meta_data, IMAGE):
                samples[sample] = data.transpose((2, 0, 1))
            else:
                raise NotImplementedError
        return samples

class ToNumpy(Transforms):
    def __init__(self, required=[], meta={}):
        super(ToNumpy, self).__init__(required, meta)

    def __call__(self, samples):
        for sample in self.required:
            data = samples[sample]
            meta_data = self.meta[sample]
            if isinstance(meta_data, IMAGE):
                samples[sample] = data.permute((0,2,3,1)).cpu().detach().numpy()
            else:
                raise NotImplementedError
        return samples
