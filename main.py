import torch
from framework.app.app import App
from framework.dataloader.DataLoader import DataLoaderController
from utils.config import Config
print('Device id: ', torch.cuda.current_device())
print('Available: ', torch.cuda.is_available())
print('Property: ', torch.cuda.get_device_properties(0))

def main(configs):
    config = None
    if isinstance(configs, list):
        config = App.make_from_config_list(config_list_paths).config
    else:
        config = App.instance(configs).config
        App.instance().update()
    dataloader_controller = DataLoaderController.instance()
    dataloader = dataloader_controller.dataloaders

    for batch in dataloader['validation']:
        img = batch['IMAGE']._data
        gt = batch['GT']._data

if __name__ == '__main__':
    config_list_paths = ['resource/configs/dataloader/dataLoaderEventQueue.yaml']
    main(config_list_paths)

a = torch.rand(5)
