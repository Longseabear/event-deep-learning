from importlib import import_module
import sys

import abc
from utils.config import Config
from collections import defaultdict


class Command(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.live = True
        if len(self.config.when.split(':')) > 1:
            self.attribute, self.step = self.config.when.split(':')
            self.step = int(self.step)
        else:
            self.attribute = self.config.when
            self.step = 1
        self.state_name = self.config.state_name if 'state_name' in self.config.keys() else None
        self.loader_name = self.config.loader_name if 'loader_name' in self.config.keys() else None
        self.config.repeat = int(self.config.repeat)

    def update(self):
        if self.config.repeat > 0:
            self.config.repeat -= 1
        if self.config.repeat == 0:
            self.live = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.config.ERROR = Exception(exc_type,exc_val, exc_tb)
            self.live = False
            raise Exception(exc_type,exc_val, exc_tb)

    def __enter__(self):
        return self

    def valid(self, controller):
        model_state = controller.get_state()
        if model_state.__getattribute__(self.attribute) % self.step != 0:
            return False
        return True

    @abc.abstractmethod
    def run(self, controller):
        raise NotImplementedError

    def leave(self):
        return self.live

class CommandSet(object):
    def __init__(self, commands):
        self.commands = commands
        self.name = commands[0].name

from .commands import *
'''
  PRINT:
    command: 'PrintCommand'
    required: []
    args: {'content':'name 1'}
    repeat: 1 # if -1, loop
    when: STEP:100 #  // START INIT STEP END TERMINATE
    target: training #(dataloader)
'''
class CommandFactory(object):
    @staticmethod
    def make_command(name, config):
        current_module = sys.modules[__name__]
        return getattr(current_module, config.command)(name, config)

class CommandController():
    def __init__(self, command_path, controller):
        self.controller = controller
        self.command_state = Config.from_yaml(command_path)
        self.command_define = self.command_state.COMMAND_DEFINE
        self.commands = defaultdict(lambda: defaultdict(list))
        self.changed = None
        self.command_objects = {}

        for command_name in self.command_state['DEFAULT']:
            self.command_parsing(command_name)

    def load_command(self, command_name):
        command_args = self.get_argument(self.command_define[command_name])
        self.command_objects[command_name] = CommandFactory.make_command(command_name, command_args)
        return self.command_objects[command_name]

    def command_parsing(self, command_name):
        if len(command_name.split("->")) > 1:
            commands_name_lists = command_name.split("->")
            command_args = self.get_argument(self.command_define[commands_name_lists[0]])
            loader_name = command_args.get('loader_name', 'global', False)
            state_name = command_args.get('state_name', 'global', False)
            self.commands[loader_name][state_name].append(CommandSet([self.load_command(c) for c in commands_name_lists]))
        else:
            command_args = self.get_argument(self.command_define[command_name])
            loader_name = command_args.get('loader_name', 'global', False)
            state_name = command_args.get('state_name', 'global', False)
            self.commands[loader_name][state_name].append(self.load_command(command_name))

    def get_argument(self, command_arg):
        arg = Config.get_empty()
        if 'parent' in command_arg.keys():
            arg = self.get_argument(self.command_define[command_arg.parent])
        arg.update(command_arg)
        return arg

    def register(self):
        pass

    def commands_run(self, commands: list):
        next_command = []
        for command in commands:
            if isinstance(command, CommandSet):
                if len(command.commands)<=0:
                    continue
                self.commands_run(command.commands)
                next_command.append(command)
                continue
            try:
                with command as c:
                    if c.valid(self.controller) and c.live:
                        out = c.run(self.controller)
                        if out is not None: self.changed.update(out)
                        c.update()
                    if command.leave(): next_command.append(command)
            except Exception as e:
                # remove them
                print(e, '[{}]', command.name)
                pass
        commands.clear()
        commands += next_command

    def run(self):
        self.changed = {}

        model_state = self.controller.get_state()
        state_name = model_state.state_name
        loader_name = model_state.loader_name

        self.commands_run(self.commands[loader_name][state_name])
        self.commands_run(self.commands[loader_name]['global'])
        self.commands_run(self.commands['global'][state_name])
        self.commands_run(self.commands['global']['global'])
        return self.changed

