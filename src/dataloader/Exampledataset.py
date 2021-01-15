from framework.dataloader.Dataset import BaseDataset
from framework.dataloader.MetaData import MetaData,ImageMeta
from skimage import io
import os
import torch

class ExampleDataset(BaseDataset):
    def __init__(self, examples=None, transform_list=[], config=None):
        super().__init__(examples, transform_list, config)

    @classmethod
    def datasetFactory(cls, config):
        dataset_folder = config['dataset_folder']

        img_paths = os.listdir(dataset_folder)
        dataset: BaseDataset = cls(config)

        dataset._examples = [] # image1_name: path, image2_name: path class
        dataset._name2metaClass = {
            'IMAGE': ImageMeta,
            'GT': ImageMeta
        }

        for img_path in img_paths:
            full_path = os.path.join(dataset_folder, img_path)
            example = {'IMAGE': full_path, 'GT': full_path}
            dataset._examples.append(example)

        return dataset

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]

        for key in example.keys():
            example[key]: MetaData = self._name2metaClass[key].load_from_item(key, example[key])
            example[key]._ToTensor()

        for transform in self._transformers:
            example = transform(example)

        return example

if __name__ == '__main__':
    required_input = ['IMAGE', "GT"]
    def make_batch(samples):
        batchs = {}
        for input_name in required_input:
            batched_data = MetaData._ToBatch(samples, input_name)
            batchs[input_name] = batched_data

        return batchs

    dataset = ExampleDataset.datasetFactory({"dataset_folder":'../../exampleDataset'})
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=make_batch)
    for batch in loader:
        img = batch['IMAGE']._data
        gt = batch['GT']._data
