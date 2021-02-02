class MainGraphFinishedException(Exception):
    def __init__(self):
        super().__init__("MainGraph Finished")
    value = property(lambda self: object(), lambda self, v: None, lambda self: None)


class MainGraphStepInterrupt(Exception):
    def __init__(self):
        super().__init__()
    value = property(lambda self: object(), lambda self, v: None, lambda self: None)


class RunnableModuleDestroyException(Exception):
    def __init__(self):
        super().__init__("This runnable module was expired.")

