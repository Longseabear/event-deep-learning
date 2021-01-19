from framework.dataloader.Dataset import BaseDataset
from framework.dataloader.TensorTypes import *
from skimage import io
import os
import torch

class ExampleDataset(BaseDataset):
    @classmethod
    def datasetFactory(cls, config):
        dataset_folder = config['dataset_folder']

        img_paths = os.listdir(dataset_folder)
        examples = [] # image1_name: path, image2_name: path class

        for i in range(50):
            for img_path in img_paths:
                full_path = os.path.join(dataset_folder, img_path)
                example = {'IMAGE': full_path, 'GT': full_path}
                examples.append(example)
        meta_data = {'IMAGE': IMAGE(), 'GT': IMAGE()}

        dataset: BaseDataset = cls(example=examples, required=config.required_input, meta=meta_data, config=config)
        return dataset

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]

        for key in self._required:
            example[key] = self._meta[key].getSample(example[key])

        for transform in self._transformers:
            example = transform(example)

        return example

if __name__ == '__main__':
    dataset = ExampleDataset.datasetFactory({"dataset_folder":'../../exampleDataset'})
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for batch in loader:
        img = batch['IMAGE']
        gt = batch['GT']
