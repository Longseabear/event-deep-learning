import threading
from contextlib import contextmanager

# ref : https://gist.github.com/tylerneylon/a7ff6017b7a1f9a506cf75aa23eacfd6#file-rwlock-py
class READ_WRITE_LOCK(object):
    def __init__(self):
        self.reader = 0

        self.write_lock = threading.Lock()
        self.reader_lock = threading.Lock()

    def read_acquire(self):
        self.reader_lock.acquire()
        self.reader += 1
        if self.reader == 1:
            self.write_lock.acquire()
        self.reader_lock.release()

    def read_release(self):
        self.reader_lock.acquire()
        self.reader -= 1
        if self.reader == 0:
            self.write_lock.release()
        self.reader_lock.release()

    def write_acquire(self):
        self.write_lock.acquire()

    def write_release(self):
        self.write_lock.release()

    @contextmanager
    def r_locked(self):
        try:
            self.read_acquire()
            yield
        finally:
            self.read_release()


    @contextmanager
    def w_locked(self):
        try:
            self.write_acquire()
            yield
        finally:
            self.write_release()
