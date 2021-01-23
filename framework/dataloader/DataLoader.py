import torch
import torch.utils.data as datalodaer
from framework.app.app import *
from framework.dataloader.Dataset import BaseDataset
from utils.runtime import get_instance_from_name, get_class_object_from_name
from abc import *
from framework.dataloader.Transform import TRANSFORM
import copy

class DataLoaderController(SingletoneInstance):
    ''':key
    DataLoaderController depends on App.
    '''
    def __init__(self):
        self.dataloader_config = App.instance().config.DATA_LOADER
        self.dataloaders = {}

#        self.make_datasets(self.dataloader_config)

    def make_dataset(self, loader_name, dataloader_config=None):
        default_loader = self.dataloader_config.required_loader['default'].copy()

        try:
            if dataloader_config is None:
                dataloader_config = default_loader.update(self.dataloader_config.required_loader[loader_name])
            else:
                dataloader_config = default_loader.update(dataloader_config)
            dataset_path = dataloader_config.dataset
            dataset_config = Config.from_yaml(dataset_path).DATASET

        except FileNotFoundError:
            print('File not founded in {}, mode:{}'.format(dataloader_config.dataset, loader_name))
            return

        if not dataloader_config.config.reload and loader_name in self.dataloaders.keys():
            return self.dataloaders[loader_name]

        dataset_module = dataset_config.module_name
        dataset_class = dataset_config.class_name
        dataset_module: BaseDataset = get_class_object_from_name(dataset_module, dataset_class)

        dataset_args = {}
        dataset_args['name'] = loader_name
        if 'dataset_args' in dataloader_config.keys():
            dataset_args = dataloader_config.dataset_args
        else:
            dataset_args = dataset_config.args[loader_name] if loader_name in dataset_config.args else \
            dataset_config.args['default']

        dataset_args['required_input'] = dataloader_config[
            'required_input'] if "required_input" in dataloader_config.keys() else dataset_args['required_input']
        dataset_args['transforms'] = dataloader_config['transforms'] if "transforms" in dataloader_config.keys() else \
        dataset_args['transforms']
        dataset = dataset_module.datasetFactory(dataset_args)

        self.dataloaders[loader_name] = self.dataloaderFactory(dataset, dataloader_config)
        return self.dataloaders[loader_name]

    def dataloaderFactory(self, dataset, config):
        class CustomDataLoader(torch.utils.data.DataLoader):
            def __init__(self, dataset, config, **kwargs):
                super(CustomDataLoader, self).__init__(dataset, **kwargs)
                self.config = config
        return CustomDataLoader(dataset, config.config, **Config.extraction_dictionary(config.loader_args))
