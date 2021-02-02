import torch
from framework.app.app import App
from framework.dataloader.DataLoader import DataLoaderController
from framework.trainer.ModelController import ModelController
from framework.ipc.ThreadCommand import MultipleProcessorController
import os

print('Device id: ', torch.cuda.current_device())
print('Available: ', torch.cuda.is_available())
print('Property: ', torch.cuda.get_device_properties(0))
print(os.environ['DISPLAY'])

def main(configs):
    config = None
    if isinstance(configs, list):
        config = App.make_from_config_list(configs).config
    else:
        config = App.instance(configs).config
        App.instance().update()
    dataloader_controller = DataLoaderController.instance()

    trainer: ModelController = ModelController.controller_factory()

    try:
        # trainer.run()
        trainer.COMMAND_CONTROLLER.run()
    except Exception as e:
        print(e)
    finally:
        MultipleProcessorController.instance().remove_all_process()

if __name__ == '__main__':
    config_list_paths = ['resource/configs/dataloader/dataLoaderEventQueue.yaml',
                         'resource/configs/trainer/ExampleController.yaml']
    main(config_list_paths)


