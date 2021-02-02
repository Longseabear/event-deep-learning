import sys
import socket
from framework.ipc.RunningCommand import *
from framework.ipc.RunningModuleVisualizer import *
from struct import unpack
from framework.app.Exceptions import *


class RunnableModuleFactory(object):
    def __init__(self, runnable_controller, host, port):
        self.runnable_controller:RunnableModuleController = runnable_controller

        self.host = host
        self.port = port
        self.client_socket = None
        self.writer_process = None

        self.job_queue = queue.Queue()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.live = True
        self.sender_thread = None
        self.receiver_thread = None

    def update_graph(self):
        if not self.job_queue.empty():
            while True:
                opt, commands = self.job_queue.get()
                opt = opt.upper()
                self.runnable_controller.get_current_main_module().write_log('[IPC receive] {} {}'.format(opt, commands))
                if opt == 'ADD':
                    self.add_from_script(commands)
                elif opt == 'REMOVE':
                    self.remove_from_script(commands)
                elif opt == 'LOAD':
                    self.update_from_yaml(commands)
                if self.job_queue.empty(): break

    def update_from_yaml(self, command_path):
        config = Config.from_yaml(command_path)
        self.runnable_controller._runnable_define.update(config.DEFINE)

        for key in config.keys():
            if key == 'DEFINE' or not isinstance(config[key], list): continue
            for script in config.get(key, [], False):
                self.job_queue.put((key, script))

    def remove_from_script(self, commands):
        try:
            m = self.runnable_controller.get_runnable_module(commands)
            m.finish = True
            m.live = False
        except RunnableModuleDestroyException as e:
            pass

    def start(self):
        threading.Thread(target=self.server_socket_run).start()
        self.sender_thread = threading.Thread(target=self.sender_run)
        self.sender_thread.start()

    def end(self):
        self.live = False
        if self.client_socket is not None:
            self.client_socket.close()
        self.server_socket.close()

        if self.receiver_thread is not None:
            self.receiver_thread.join()
        if self.sender_thread is not None:
            self.sender_thread.join()

    def __del__(self):
        self.end()

    def server_socket_run(self):
        while self.live:
            self.client_socket = None
            self.server_socket.listen()
            try:
                self.client_socket, addr = self.server_socket.accept()
                self.receive()
            except Exception as e:
                print(e)
                self.client_socket.shutdown(socket.SHUT_WR)
                self.client_socket.close()

    def receive(self):
        while self.live:
            byte_len = self.client_socket.recv(8)
            (length,) = unpack('>Q', byte_len)
            buffer = b''
            while len(buffer) < length:
                # doing it in batches is generally better than trying
                # to do it all in one go, so I believe.
                to_read = length - len(buffer)
                buffer += self.client_socket.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            self.client_socket.sendall(b'\00')

            data = buffer.decode().split(' ', maxsplit=1)
            if len(data) <= 1:
                continue
            self.job_queue.put((data[0].strip(), data[1].strip()))

    def sender_run(self):
        while self.live:
            self.sender()
            time.sleep(1)

    def sender(self):
        workspace = self.runnable_controller.config.get('save_path', '$base/ipc', possible_none=False)
        saved_path = App.instance().dir_path_parser(os.path.join(workspace, 'monitor.txt'))
        with open(saved_path, 'w') as f:
            f.write("{} {}\n".format(self.host, self.port))
            paths = DisplayRunnableModule.make_tree(self.runnable_controller, 'root')
            f.writelines([path.displayable(self.runnable_controller) + '\n' for path in paths])

    @staticmethod
    def parsing(line, separator='->', opener='(', closer=')'):
        res = []
        count = 0
        temp = ""
        for c in line:
            if c == opener: count += 1
            elif c == closer: count -= 1

            if count == 0 and c == separator[0]:
                res.append(temp)
                temp = ''
            elif count == 0 and c in separator:
                pass
            else:
                temp += c
        if len(temp) > 0: res.append(temp)
        return res

    def connect_edge(self, a, b):
        self.runnable_controller.next[a].append(b)
        self.runnable_controller.get_runnable_module(b).indegree += 1
        self.runnable_controller.previous[b].append(a)

    def create_module(self, module_name):
        if module_name[0] == '$':
            if module_name[1:] in self.runnable_controller.all_runnable_module.keys():
                return module_name[1:]
            return self.create_runnable_module(module_name[1:], self.runnable_controller._runnable_define[module_name[1:]])
        elif module_name[0] == '*':
            return self.create_runnable_module(module_name[1:], RunnableModule.get_default_config(module_name[1:]))
        else:
            return self.create_runnable_module(module_name, self.runnable_controller._runnable_define[module_name])

    def module_parser(self, name):
        name = name.strip()
        if '(' in name:
            modules = [i for i in re.split(r'\((.*)\)', name) if len(i) > 0]
            if len(modules)==1:
                return self.add_from_script(modules[0])
            elif len(modules)==2:
                sub_names = self.add_from_script(modules[1])
                name = self.create_module(modules[0])
                self.runnable_controller.all_runnable_module[name].required.update(set(sub_names))
                return [name]
            else:
                raise NotImplementedError("{}".format(modules))
        else:
            return [self.create_module(name)]

    def add_from_script(self, script):
        runnable_modules = [i.strip() for i in self.parsing(script)]
        module_names = []
        pre = None
        for module in runnable_modules:
            outs = self.module_parser(module)
            module_names += outs
            if pre is not None:
                self.connect_edge(pre, outs[0])
            pre = outs[-1]
        return module_names

    def create_runnable_module(self, name, config):
        if name in RunnableModule.get_cls_count().keys():
            RunnableModule.get_cls_count()[name] += 1
            name = "{}_{}".format(name, RunnableModule.get_cls_count()[name])
        else:
            RunnableModule.get_cls_count()[name] = 0

        current_module = sys.modules[__name__]
        config = self.get_argument(config)

        self.runnable_controller.all_runnable_module[name] = getattr(current_module, config.command)(name, config)
        return name

    def get_argument(self, command_arg):
        """
        [Configuration safe]
        Create a new argment that inherits the parent property of argmemt.
        :param command_arg:
        :return:
        """
        arg = Config.get_empty()
        if 'inherit' in command_arg.keys():
            arg = self.get_argument(self.runnable_controller._runnable_define[command_arg.inherit])
        arg.update(command_arg)
        return arg

class RunnableModuleController(object):
    def __init__(self, config, controller):
        self.config = config

        self.next = defaultdict(list)
        self.previous = defaultdict(list)

        self.make_directory()
        self.factory = RunnableModuleFactory(self, config.ipc_host, int(config.ipc_port))
        RunnableModule.set_global_module(controller, self)

        self.command_configs = Config.from_yaml(self.config.command_path)
        self._runnable_define = self.command_configs.DEFINE

        # Shared memory: self.variables
        self.variables = defaultdict(None)

        self.variables['main'] = ['root']
        self.variables['iter'] = ['root']
        self.variables['self'] = ['root']

        root_config = RunnableModule.get_default_config('RunnableGraph')
        root_config.repeat = 1
        self.root: RunnableModule = RunnableGraph(config=root_config, name='root')
        self.all_runnable_module = {'root': self.root}

        for script in self.command_configs.DEFAULT:
            self.factory.job_queue.put(('ADD', script))

        self.factory.start()
        #DEBUG
        # for key in self.all_runnable_module.keys():
        #     print(key, self.all_runnable_module[key].indegree)
        #     print(self.all_runnable_module[key].required)
        #     print('adj', self.next[key])
        # input()

    def get_runnable_module(self, key):
        """
        :param key:
        This function supports variable type arguments.
        In the case of $, the value is taken from self.vairables and used as a key.
        If self.variables is a list, select the last value.
        :return:
        """
        try:
            if key[0]=='$':
                key = self.get_last_variables_value(key[1:])
            return self.all_runnable_module[key]
        except AttributeError:
            return None
        except KeyError:
            raise RunnableModuleDestroyException()

    def remove_runnable_module(self, key):
        del self.all_runnable_module[key]

    def make_directory(self):
        workspace = self.config.get('save_path', '$base/ipc', possible_none=False)
        App.instance().make_save_dir(workspace)

    def get_last_variables_value(self, name):
        item = self.variables[name]
        if isinstance(item, list):
            item = item[-1]
        return item

    def get_current_main_module(self):
        """
        return main module.
        The object of this function must be used only as a local variable.
        """
        return self.get_runnable_module(self.variables['main'][-1])

    def run(self):
        def loading():
            while True:
                for i in range(4): yield '.' * i

        temp_bar = loading()
        while True:
            try:
                self.root.step = 0
                self.root.run()
                while len(self.root.required)==0 and self.factory.job_queue.empty():
                    print('\ridle' + next(temp_bar), end='')
                    time.sleep(0.5)
            except Exception as e:
                if App.instance().config.App.DEBUG:
                    print(traceback.format_exc())
                self.factory.end()
                raise e

