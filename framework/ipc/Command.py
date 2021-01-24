from importlib import import_module
import sys

import abc
from utils.config import Config
from collections import defaultdict
import traceback

class Command(object):
    def __init__(self, name, config):
        """
        Command Class
        Command class can be used with the with statement.
        :param name:
        :param config:
        """
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
            self.config.ERROR = Exception(exc_type, exc_val, exc_tb)
            self.live = False
            raise Exception(exc_type, exc_val, exc_tb)

    def __del__(self):
        self.destroy()

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

    def destroy(self):
        pass

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
        """
        Controll all commands.

        self.controller: main controller reference
        self.command_state: command configuration. The configuration operates independently from the upper class command controller.
        that is, it is Configuration safe
        self._command_define: READONLY dictionary. note that command define cannot be modified except for the controller.
        self.commands: It is responsible for the life cycle of all commands. It is composed of a two-dimensional dictionary and is accessed with [LOADER_NAME][STATE_NAME].
        If you want it to be done in all cases, use the'global' keyword.
        self.chaned: Cache the return value.

        :param command_path: yaml file,
        default-> MODEL_CONTROLLER.COMMAND_CONTROLLER.command_path: 'resource/ipc/default_command.yaml'

        :param controller: controller
        """
        self.controller = controller
        self.command_state = Config.from_yaml(command_path)
        self.commands = defaultdict(lambda: defaultdict(list))
        self.changed = None
        self._command_define = self.command_state.COMMAND_DEFINE

        for command_name in self.command_state['DEFAULT']:
            self.command_parsing(command_name)

    @property
    def command_define(self):
        return self._command_define

    def load_command(self, command_name: str) -> Command:
        """
        load command using command_name.
        get_argument function is configuration safe. Therefore, this function is also configuration safe.

        :param command_name: All command information is recorded in the command yaml file.
        :return: Command object
        """
        command_args = self.get_argument(self.command_define[command_name])
        return CommandFactory.make_command(command_name, command_args)

    def command_parsing(self, command_name: str):
        """
        Command: Typical single command.
        CommandSet: A command with multiple sequences. Separate with '->'.
                    In the case of the command argument, the point of invocation depends on the first command.
                    So, if you want to make the call point whole, use DummyCommand first.

        :rtype: object
        """
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
        """
        [Configuration safe]
        Create a new argment that inherits the parent property of argmemt.
        :param command_arg:
        :return:
        """
        arg = Config.get_empty()
        if 'parent' in command_arg.keys():
            arg = self.get_argument(self.command_define[command_arg.parent])
        arg.update(command_arg)
        return arg

    def register(self):
        pass

    def commands_run(self, commands: list):
        """
        Command run operates in the following cycle.
        1. Take out commands in the order they came in
        2. Check if it is valid, and also check if it is alive/
        3. If 2 is true, run.
        4. Execute update()
        5. leave() -> If true, keep command again
        6. If false, destory()

        Note that CommandController cannot change the current state.
        If the if command modifies the current state, it won't work.

        destroy() is called whenever Command is destroyed.
        It can also be removed by remote ipc.


        :param commands:

        """
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
                    else: command.destroy()

            except Exception as e:
                if App.instance().config.App.DEBUG:
                    print(traceback.format_exc())
                else:
                    print('[ERROR/{}]'.format(c.command))
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

