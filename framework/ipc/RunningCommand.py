import matplotlib.pyplot as plt
import copy
from framework.app.Format import *
from collections import defaultdict
from framework.ipc.ThreadCommand import *
from utils.config import Config
from framework.dataloader.DataLoader import DataLoaderController
import traceback
from framework.app.Exceptions import *
from tqdm import tqdm
import os
from framework.dataloader.Transform import TRANSFORM
from framework.dataloader.TensorTypes import *


class DependentParent(object):
    def __init__(self, my, run_cycle: str):
        abb = run_cycle.split(':')
        default_step = 1
        if len(abb) > 1:
            run_cycle = abb[0]
            default_step = int(abb[1])

        cut = run_cycle.split('.', maxsplit=1)
        if len(cut)==1: cut.append("step")

        self.my = my
        self.dependent_module, self.attribute = cut
        self.requried_step = default_step
        self.previous_step = -1

    def get_dependent_module(self):
        return self.my.get_runnable_controller().get_runnable_module(self.dependent_module)

    def get_condition_value(self):
        return self.my.get_runnable_controller().get_runnable_module(self.dependent_module).__getattribute__(self.attribute)

    def valid_check(self):
        now_step = self.get_condition_value()
        if now_step % self.requried_step == 0 and self.previous_step != now_step and self.my.live:
            self.previous_step = now_step
            return self.my.repeat == -1 or self.my.repeat > self.my.step
        return False

class RunnableModule(object):
    __cls_counter = defaultdict(int)
    __model_controller = None
    __runnable_controller = None
    __factory = None
    __global_count = 0

    @classmethod
    def get_default_config(cls, command_name):
        args = {}
        args['command'] = command_name
        args['required'] = []
        args['repeat'] = 1
        args['args'] = {}
        return Config.from_dict(args)

    @property
    def factory(self):
        return RunnableModule.__factory

    @classmethod
    def set_global_module(cls, model_controller, runnable_controller):
        cls.__model_controller = model_controller
        cls.__runnable_controller = runnable_controller
        cls.__factory = runnable_controller.factory

    @classmethod
    def get_cls_count(cls):
        return cls.__cls_counter

    @classmethod
    def get_model_controller(cls):
        return cls.__model_controller

    @classmethod
    def get_runnable_controller(cls):
        return cls.__runnable_controller

    def write_log(self, contents, **kwargs):
        latest_state_obj = self.get_runnable_controller().get_current_main_module()
        if 'tqdm' not in dir(latest_state_obj) or latest_state_obj is None:
            print(contents, **kwargs)
            return
        if latest_state_obj.tqdm is None:
            print(contents, **kwargs)
        else:
            latest_state_obj.tqdm.write(contents)

    def __init__(self, name, config):
        """

        :param name:
        :param config: must to be configuration safe
        :param generation:
        :param parent:
        """
        self.config = config
        self.required = set(self.config.get('required', []))

        self.name = name
        self.step = 0
        self.finish = self.config.get('finish', False)
        self.live = True

        # Graph Module
        self.indegree = 0
        self.current_degree = 0

        self.write_log("[COMMAND JOIN] ModuleName:{}, Command:{}".format(self.name, self.__class__.__name__))
        """
        Command Class
        Command class can be used with the with statement.
        :param name:
        :param config:
        """
        self.dependent = DependentParent(self, self.config.get('run_cycle', "$self.step:1"))
        self.repeat = int(self.config.get('repeat', 1, False))
        if not 'args' in self.config.keys():
            self.config.args = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.current_degree = 0

        if exc_type is not None:
            self.config.ERROR = Exception(exc_type, exc_val, exc_tb)
#            raise Exception(exc_type, exc_val, exc_tb)

    def __enter__(self):
        return self

    def start(self):
        """
        Must be executed before the module is executed.
        """
        RunnableModule.get_runnable_controller().variables['self'].append(self.name)

    def valid(self):
        return self.dependent.valid_check()

    def run(self):
        """
        Must be executed before the module is executed.
        """
        raise NotImplementedError

    def __del__(self):
        self.write_log("[COMMAND EXPIRED] ModuleName:{}, Command:{}".format(self.name, self.__class__.__name__))

    def update(self):
        """
        Must be executed before the module is executed.
        """
        raise NotImplementedError

    def end(self):
        self.current_degree = 0

        if RunnableModule.get_runnable_controller().variables['self'][-1] == self.name:
            RunnableModule.get_runnable_controller().variables['self'].pop()

        if self.finish and (self.repeat != -1 and self.repeat <= self.step):
            self.live = False

    def leave(self):
        return self.live

    def destroy(self):
        raise NotImplementedError

class RunnableGraph(RunnableModule):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config)
        self.process_queue = queue.Queue()

    def start(self):
        super(RunnableGraph, self).start()

    def end(self):
        super(RunnableGraph, self).end()

    def init(self):
        self.process_queue = queue.Queue()
        self.step += 1

        # UPDATE GRAPH
        self.factory.update_graph()
        for obj_name in self.required:
            obj = RunnableModule.get_runnable_controller().get_runnable_module(obj_name)
            obj.step = 0
            obj.current_degree = 0
            if obj.indegree == 0:
                self.process_queue.put(obj_name)

    def update(self):
        pass

    def destroy(self):
        module_controller = RunnableModule.get_runnable_controller()
        for obj_name in self.required:
            module_controller.get_runnable_module(obj_name).destroy()
        module_controller.remove_runnable_module(self.name)

    def remove_edge(self, name):
        module_controller = RunnableModule.get_runnable_controller()
        next_names = module_controller.next[name]
        previous_names = module_controller.previous[name]

        for next_name in next_names:
            next_obj = module_controller.get_runnable_module(next_name)
            next_obj.indegree -= 1
            next_obj.indegree += len(previous_names)
            next_obj.current_degree += len(previous_names)

            module_controller.previous[next_name].remove(name)
            module_controller.previous[next_name] += previous_names
            if next_obj.current_degree == next_obj.indegree:
                self.process_queue.put(next_name)

        for pname in previous_names:
            module_controller.next[pname].remove(name)
            module_controller.next[pname] += next_names

        del module_controller.next[name]
        del module_controller.previous[name]
        self.required.remove(name)

    def process(self, obj_name):
        module_controller = RunnableModule.get_runnable_controller()
        try:
            with module_controller.get_runnable_module(obj_name) as p:
                if p.valid():
                    p.start()
                    p.run()
                    p.update()
        except MainGraphStepInterrupt as e:
            if isinstance(self, MainGraph):
                pass
            else:
                p.end()
                if not p.leave():
                    self.remove_edge(p.name)
                    p.destroy()
                raise e
        except MainGraphFinishedException as e:
            p.end()
            if not p.leave():
                self.remove_edge(p.name)
                p.destroy()
            raise e
        except Exception as e:
            if App.instance().config.App.DEBUG:
                print(traceback.format_exc())
            else:
                self.write_log('[ERROR/{}/{}] {}'.format(p.name, p.__class__.__name__, e))
            pass
        finally:
            p.end()
            if not p.leave():
                self.remove_edge(p.name)
                p.destroy()
            else:
                nexts_name = module_controller.next[p.name]
                for name in nexts_name:
                    next_obj = module_controller.get_runnable_module(name)
                    next_obj.current_degree += 1
                    if next_obj.current_degree == next_obj.indegree:
                        self.process_queue.put(name)

    # Must to SCC
    def run(self):
#        self.write_log("[START] -> {} {}/{}, task={}".format(self.name, self.step, self.config.repeat, self.required))
        while self.valid():
            self.init()
#            self.write_log("[PROCESSING] -> {} {}/{}, task={}".format(self.name, self.step, self.config.repeat, self.required))
            while not self.process_queue.empty():
                obj_name: str = self.process_queue.get()
                self.process(obj_name)
#        self.write_log("[END] -> {} {}/{}".format(self.name, self.step, self.config.repeat))

class MainGraph(RunnableGraph):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config, **kwargs)
        self.iter = None
        self.tqdm = None

        self.total_step = 0
        self.loader_name = self.config.args.loader_name

    def get_epoch(self):
        return self.step - 1

    def start(self):
        super(MainGraph, self).start()
        RunnableModule.get_runnable_controller().variables['main'].append(self.name)

        config = copy.deepcopy(self.config.args[self.loader_name]) if self.loader_name in self.config.args.keys() else None
        DataLoaderController.instance().make_dataset(self.loader_name, config)
    
    def init(self):
        super(MainGraph, self).init()
        self.tqdm = tqdm(DataLoaderController.instance().dataloaders[self.loader_name],
                                       bar_format='{l_bar}{bar:10}{r_bar}', ascii=True)
        self.tqdm.desc = '[{}] {}/{} '.format(self.loader_name, self.step, self.repeat)
        self.iter = iter(self.tqdm)

    def destroy(self):
        super(MainGraph, self).destroy()
        self.tqdm = None
        self.iter = None

    def end(self):
        super(MainGraph, self).end()
        if RunnableModule.get_runnable_controller().variables['main'][-1] == self.name:
            RunnableModule.get_runnable_controller().variables['main'].pop()

    def run(self):
        try:
            super(MainGraph, self).run()
        except MainGraphFinishedException as e:
            self.write_log('[INTERRUPT/{}/{}] {}'.format(self.name, self.__class__.__name__, e))

# -------------------------------- NODE
class RunnableNode(RunnableModule):
    def __init__(self, name, config):
        super().__init__(name, config)

    def start(self):
        pass

    def run(self):
        raise NotImplementedError

    def update(self):
        self.step += 1

    def end(self):
        super(RunnableNode, self).end()

    def destroy(self):
        module_controller = RunnableModule.get_runnable_controller()
        module_controller.remove_runnable_module(self.name)

class TestCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        new_step = int(self.config.args.new_step)
        obj = self.get_runnable_controller().get_current_main_module()
        previous_step = obj.step
        obj.step = new_step
        self.write_log('previous step: {} -> new step {}'.format(previous_step, new_step))
        if not obj.valid():
            raise MainGraphFinishedException

class PrintCommand(RunnableNode):
    def __init__(self, name, config):
        config = Config.from_dict({
            'args':{
                'content': 'default contents'
            }
        }).update(config)
        super().__init__(name, config)

    def run(self):
        self.write_log('[{}] PRINT MODULE: {} step: {} / repeat: {}'.format(self.name, self.config.args.content,
                                                                            self.dependent.get_dependent_module().__getattribute__(self.dependent.attribute), self.repeat), end=' ')
        for name in self.required:
            self.write_log(name, end=" ")
#            self.write_log(name + " " + controller.sample[name])

class PrintStateCommand(RunnableNode):
    def __init__(self, name, config):
        config = Config.from_dict({
            'args':{
                'obj': '$main'
            }
        }).update(config)
        super().__init__(name, config)

    def run(self):
        obj = self.get_runnable_controller().get_runnable_module(self.config.args.obj)
        for name in self.required:
            self.write_log('[{}.{}]: {} / '.format(obj.name, name, obj.__getattribute__(name)), end=' ')
        self.write_log('')

class RunCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        for name in self.required:
            if name in self.get_model_controller().all_callable:
                self.get_model_controller().__getattribute__(name)(**self.config.args)

class BatchedImageShowCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name + "_output" for name in self.config.required], {'ALL': IMAGE()})

    def run(self):
        temp_sample = {}
        for name in self.config.required:
            temp_sample[name + "_output"] = self.get_model_controller().sample[name]
        temp_sample = self.numpy_trasnsform(temp_sample)

        for key in temp_sample.keys():
            plt.imshow(temp_sample[key][self.config.args.batch_number])
            plt.show()

class BatchedImageSaveCommand(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name for name in self.config.required], {'ALL': IMAGE()})
        self.type = IMAGE()

    def leave(self):
        if self.live or not MultipleProcessorController.instance().finished(self.__class__.__name__):
            return True
        else:
            return False

    def destroy(self):
        MultipleProcessorController.instance().remove_process(self.__class__.__name__)

    def run(self):
        args = []
        dir_path = App.instance().dir_path_parser(self.config.args.get('path', '$base/visual/img', possible_none=False))
        fm = self.config.args.get('format', 'png', possible_none=False)

        App.instance().make_save_dir(dir_path)
        formatter = MainStateBasedFormatter(self.get_runnable_controller(), {'content': '', 'format': fm, 'batch':0},
                              format='[$main:step:03]e_[$main:total_step:08]s_[$content]_[$batch].[$format]')

        for name in self.config.required:
            formatter.contents['content'] = name
            imgs = self.numpy_trasnsform({name:self.get_model_controller().sample[name].clone()})

            b,_,_,_ = imgs[name].shape
            for i in range(b):
                formatter.contents['batch'] = str(i).zfill(4)
                path = os.path.join(dir_path, formatter.Formatting())
                args.append((path, imgs[name][i]))

        import time
        def batched_image_save(queue):
            while True:
                sample = queue.get()
                if sample is None: break
                path, img = sample
                misc.imsave(path, img)
                time.sleep(0.001)
        MultipleProcessorController.instance().push_data(self.__class__.__name__, batched_image_save, args, num_worker=1)


class ModuleLoadClass(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        for name in self.required:
            args = Config.from_dict({
                'module_name': name,
                'controller': self.get_runnable_controller(),
                'path': '$latest',
                'load_strict': True
            })
            args.update(self.config.args[name])
            try:
                if args['path'] == '$latest':
                    args['path'] = args['path'] + '_{}'.format(name)
                args['path'] = App.instance().dir_path_parser(args['path'])
            except KeyError as e:
                self.write_log('[ERROR] App.variables load error. key:{}.'.format(args['path']))
                continue

            if name not in dir(self.get_model_controller()):
                self.write_log('[ERROR] Load fail: {} is not Modules.'.format(name))
                continue

            module = self.get_model_controller().__getattribute__(name)
            if not isinstance(module, torch.nn.Module):
                self.write_log('[ERROR] Load fail: {} is not Modules'.format(name))
                continue
            if not callable(getattr(module, 'load')):
                self.write_log('[ERROR] Load fail: {} must to have load method.'.format(name))
                continue
            module.load(args)
            self.write_log('[INFO] {} Load scucesses [{}] '.format(self.config.required, args['path']))

class ModuleSaveClass(RunnableNode):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self):
        for name in self.required:
            if name not in dir(self.get_model_controller()):
                self.write_log('[ERROR] save fail: {} is not Modules.'.format(name))
                continue
            args = Config.from_dict({
                'module_name': name,
                'controller': self.get_runnable_controller(),
                'path': '$base/ckpt_{}'.format(name),
                'load_strict': True
            })
            args.update(self.config.args[name])
            args['path'] = App.instance().dir_path_parser(args['path'])

            module = self.get_model_controller().__getattribute__(name)
            if not isinstance(module, torch.nn.Module):
                self.write_log('[ERROR] save fail: {} is not Modules'.format(name))
                continue
            if not callable(getattr(module, 'save')):
                self.write_log('[ERROR] save fail: {} must to have save method.'.format(name))
                continue

            module.save(args)
            self.write_log('[INFO] {} Save scucesses [{}] '.format(self.config.required, App.instance().get_variables('$latest_{}'.format(name))))


