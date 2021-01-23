import torch
import scipy.misc as misc

class TensorType:
    def __str__(self):
        return self.__class__.__name__

    def getSample(self, sample):
        raise NotImplementedError

class IMAGE(TensorType):
    def image_loader(self, path):
        return misc.imread(path)/255.

    def image_saver(self, path, data):
        return misc.imsave(path, data)

    def getSample(self, sample):
        return self.image_loader(sample)

class GRAYSCALE_IMAGE(IMAGE):
    def image_loader(self, path):
        return misc.imread(path)

    def getSample(self, sample):
        return self.image_loader(sample)


class DISPARITY(TensorType):
    pass

