

class Command(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    