import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from framework.app.app import App
from utils.runtime import get_instance_from_name, get_class_object_from_name
from utils.config import Config
import os

def make_optimizer(args, model):
    '''
    ref: https://github.com/thstkdgus35/EDSR-PyTorch/blob/master/src/utility.py
    '''

    trainable = filter(lambda x: x.requires_grad, model.parameters())
    optimizer_class = get_class_object_from_name(args.optimizer.optimizer_module,
                                                 args.optimizer.optimizer_class)
    optimizer_args = Config.extraction_dictionary(args.optimizer.optimizer_args)

    scheduler_class = get_class_object_from_name(args.scheduler.scheduler_module,
                                       args.scheduler.scheduler_class,)
    scheduler_args = Config.extraction_dictionary(args.scheduler.scheduler_args)

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, info):
            dst = info['path']
            file_names = os.listdir(dst)
            previous_file_name = None
            for name in file_names:
                _, file_extension = os.path.splitext(name)
                if file_extension == 'opt':
                    epoch = name.split('_')[-2]
                    if int(epoch) == info['state'].epoch:
                        previous_file_name = name

            torch.save(self.state_dict(), os.path.join(dst, self.get_save_name(info['state'])))
            if previous_file_name is not None:
                os.remove(os.path.join(dst, previous_file_name))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

        def get_save_name(self, state):
            return App.instance().name_format(App.instance().name) + "_{}_{}.opt".format(state.epoch, state.step)

    optimizer = CustomOptimizer(trainable, **optimizer_args)
    optimizer._register_scheduler(scheduler_class, **scheduler_args)
    return optimizer
