import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
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

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **optimizer_args)
    optimizer._register_scheduler(scheduler_class, **scheduler_args)
    return optimizer
