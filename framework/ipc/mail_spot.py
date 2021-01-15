from framework.app.app import SingletoneInstance
from .Command import *
import os

'''file_Example.yaml(yaml)
info:
    command_step: 200
    
per_epoch:
    
one_per_step:
    
'''
class CommandRequestResource(object):
    def __enter__(self, path):
        self.f = open(path, 'r')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def request(self):
        commands = self.f.readlines()

class MailSpot(SingletoneInstance):
    def __init__(self, config):
        self.config = config
        self.commands = []
        if not os.path.exists(self.config.request_path):
            with open(self.config.request_path, 'w') as f:
                pass

        if not os.path.exists(self.config.answer_path):
            with open(self.config.answer_path, 'w') as f:
                pass

    # read only
    def request(self):
        with open(self.config.request_path) as f:

    # write only
    def answer(self):
