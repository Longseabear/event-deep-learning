import torch
import torch.utils.data as datalodaer
from framework.app.app import *
from framework.dataloader.Dataset import BaseDataset
from framework.dataloader.MetaData import MetaData
from utils.runtime import get_instance_from_name, get_class_object_from_name
from abc import *

class DataLoaderController(SingletoneInstance):
    ''':key
    DataLoaderController depends on App.
    '''
    def __init__(self):
        self.dataloader_config = App.instance().config.DATA_LOADER
        self.dataloaders = {}

        args = self.dataloader_config

        for loader_name in args.required_loader.keys():
            try:
                dataset_path = args.required_loader[loader_name].dataset
                dataset_config = Config.from_yaml(dataset_path).DATASET
            except FileNotFoundError:
                print('File not founded in {}, mode:{}'.format(dataset_path, loader_name))
                continue
            dataset_module = dataset_config.module_name
            dataset_class = dataset_config.class_name
            dataset_module: BaseDataset = get_class_object_from_name(dataset_module,dataset_class)


            dataset_args = dataset_config.args[loader_name] if loader_name in dataset_config.args else dataset_config.args['default']
            dataset = dataset_module.datasetFactory(dataset_args)

            self.dataloaders[loader_name] = self.dataloaderFactory(dataset, loader_name)

    def dataloaderFactory(self, dataset, mode):
        return torch.utils.data.DataLoader(dataset, collate_fn=self.make_collate_fun(), **Config.extraction_dictionary(self.dataloader_config.required_loader[mode].loader_args))

    def make_collate_fun(self):
        def make_batch(samples):
            batchs = {}
            for input_name in DataLoaderController.instance().dataloader_config['required_input']:
                batched_data = MetaData._ToBatch(samples, input_name)
                batchs[input_name] = batched_data
            return batchs
        return make_batch
