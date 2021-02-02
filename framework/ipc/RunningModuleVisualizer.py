from pathlib import Path
from framework.ipc.RunningCommand import RunnableGraph
from queue import Queue

class DisplayRunnableModule(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, name, parent_path, is_last):
        """
        #https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

        :param path:
        :param parent_path:
        :param is_last:
        """
        self.name = name
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @staticmethod
    def get_ordered_module(controller, base):
        children = controller.get_runnable_module(base).required
        outs = []
        indegree = {}
        q = Queue()
        for name in children:
            indegree[name] = int(controller.get_runnable_module(name).indegree)
            if indegree[name] == 0:
                q.put(name)

        while not q.empty():
            cur_name = q.get()
            outs.append(cur_name)
            for next_name in controller.next[cur_name]:
                indegree[next_name]-=1
                if indegree[next_name]==0:
                    q.put(next_name)
        return outs

    @classmethod
    def make_tree(cls, controller, root, parent=None, is_last=False):
        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = cls.get_ordered_module(controller, root)
        count = 1
        for path in children:
            is_last = count == len(children)
            if isinstance(controller.get_runnable_module(path), RunnableGraph):
                yield from cls.make_tree(controller, path,
                                         parent=displayable_root,
                                         is_last=is_last)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    def display_name(self, controller):
        if isinstance(controller.get_runnable_module(self.name), RunnableGraph):
            return self.name + '/'
        return self.name

    def displayable(self, controller):
        if self.parent is None:
            return self.display_name(controller)

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.display_name(controller))]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))
