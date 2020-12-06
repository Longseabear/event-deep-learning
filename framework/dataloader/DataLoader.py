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
        self.dataset_config = App.instance().config.DATASET
        self.dataset_module: BaseDataset = get_class_object_from_name(self.dataset_config.module_name,
                                                 self.dataset_config.class_name)

    #Override
    @classmethod
    def dataloaderFactory(cls, dataset, mode=''):
        dataloader_config = cls.instance().dataloader_config
        return torch.utils.data.DataLoader(dataset, collate_fn=cls.make_collate_fun(), **dataloader_config.args[mode].loader_args)

    @classmethod
    def make_collate_fun(self):
        def make_batch(samples):
            batchs = {}
            for input_name in DataLoaderController.instance().dataloader_config['required_input']:
                batched_data = MetaData._ToBatch(samples, input_name)
                batchs[input_name] = batched_data

            return batchs
        return make_batch

    @classmethod
    def splitFactory(cls):
        ''':key
        A function that separates samples into training, testing, and validation
        reurn_type: utils.data.DataLoader
        '''

        controller = cls.instance()
        args = controller.dataloader_config

        dataloaders = {}
        for loader_name in args.required_loader:
            dataset = controller.dataset_module.datasetFactory(
                **Config.extraction_dictionary(controller.dataset_config.args[loader_name]))
            dataloaders[loader_name] = cls.dataloaderFactory(dataset, loader_name)

        return dataloaders
