import torch
from framework.app.app import App
from framework.dataloader.DataLoader import DataLoaderController
from framework.trainer.ModelController import ModelController
from utils.config import Config
from framework.trainer.ModelController import ModelState

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

    trainer = ModelController.controller_factory()
    while True:
        print(trainer.optimizer.get_last_epoch())
        trainer.set_state(ModelState(loader_name='training'))
        trainer.train()

if __name__ == '__main__':
    config_list_paths = ['resource/configs/dataloader/dataLoaderEventQueue.yaml',
                         'resource/configs/trainer/ExampleController.yaml']
    main(config_list_paths)