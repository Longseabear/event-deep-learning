import torch.utils.data as datalodaer
from framework.dataloader.Transform import TRANSFORM
from abc import *


class BaseDataset(datalodaer.Dataset, metaclass=ABCMeta):
    def __init__(self, example: list, required: list, meta, config):
        """
        [configuration unsafe]
        Dataset Class works independently from App Class Configuration. So, it doesn't matter if the configuration is unsafe.

        :param example: The example must have information that can load all sample data. e.g., path information
        :param required: required_input
        :param meta: meta defines TensorTypes of the required data. When the class is inherited, the type for internal information must be specified.
        :param config: The dataset class is independent of the App. Therefore, it does not matter if it is corrected.
        """
        self._examples: list = example
        self._required: list = required
        self._meta = meta
        self.config = config
        self._transformers = []
        self.make_transforms()

    def make_transforms(self):
        """
        Create a transform method based on the transform recorded in the configuration.
        format: [TransformName:INPUT1-INPUT2-...]
        e.g., [ToTensor:IMAGE-GT]
        """
        for transforms_info in self.config.transforms:
            transforms_info = transforms_info.split(':')
            transform_name = transforms_info[0]
            required_inputs = set(self._required)
            if len(transforms_info) > 1:
                required_inputs = required_inputs.intersection(set(transforms_info[1].split('-')))
            self._transformers.append(TRANSFORM[transform_name](list(required_inputs), self._meta))

    @classmethod
    @abstractmethod
    def datasetFactory(self, config):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

