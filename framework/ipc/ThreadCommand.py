import threading
import queue
from framework.ipc.Command import Command
from framework.app.app import SingletoneInstance
from multiprocessing import Queue, Process
import time

class ProcessWrapper(object):
    def __init__(self, func, worker_num=8):
        self.queue = Queue()
        self.worker_num = worker_num
        self.processes = [Process(target=func, args=(self.queue,)) for _ in range(self.worker_num)]

    def run(self):
        for p in self.processes: p.start()

    def stop(self):
        for _ in range(self.worker_num): self.queue.put(None)
        while not self.queue.empty(): time.sleep(0.5)
        for p in self.processes: p.join()

    def __del__(self):
        print('ProcessWarpper del')

class MultipleProcessorController(SingletoneInstance):
    def __init__(self):
        self.processes = {}

    def remove_all_process(self):
        for name in self.processes.keys():
            p = self.processes.get(name, None)
            if p is None: continue
            self.remove_process(name)

    def finished(self, name):
        p = self.processes.get(name, None)
        if p is None:
            return True
        return self.processes[name].queue.empty()

    def remove_process(self, name):
        p = self.processes.get(name, None)
        if p is None:
            return
        p.stop()
        self.processes[name] = None

    def make_new_process(self, name, func, num_worker=8):
        self.processes[name] = ProcessWrapper(func, num_worker)

    def push_data(self, name, func, datas, num_worker=8):
        if self.processes.get(name, None) is None:
            self.make_new_process(name, func, num_worker=num_worker)
            self.processes[name].run()

        process: ProcessWrapper = self.processes[name]
        for data in datas:
            process.queue.put(data)