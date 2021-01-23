class DataLoaderIteration(object):
    def __init__(self, loader):
        self.iter = iter(loader)



a = hn_wrapper(range(1,5))
while a.hasnext():
    print(a.next())