import numpy as np
from Queue import Queue
import threading
from time import time

class BatchIterator(object):

    def __init__(self, file_dir, label_file, batch_size, process_func=None, testing=None):
        self.files = data.get_image_files(file_dir)
        names = data.get_names(files)
        self.labels = data.get_labels(names, label_file=label_file).astype(np.float32)
        self.n = len(files)
        self.batch_size = batch_size
        self.testing = testing

        if process_func is None:
            process_func = lambda x: x
        self.process_func = process_func

        if not self.testing:
            self.create_index = lambda: np.random.permutation(self.n)
        else:
            self.create_index = lambda: range(self.n)

        self.indices = self.create_index()
        assert self.n > self.batch_size


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_permuted_batch_idx(self):
        if len(self.indices) <= self.batch_size:
            new_idx = self.create_index()
            self.indices = np.hstack([self.indices, new_idx])

        batch_idx = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]

        return batch_idx

    def next(self):
        batch_idx = self.get_permuted_batch_idx()
        batch_files = self.files[batch_idx]
        batch_X = data.load_image(batch_files)
        batch_X = self.process_func(batch_X)
        batch_y = self.labels[batch_idx]
        return (batch_X, batch_y)

def threaded_iterator(iterator, num_cached=50):
    queue = Queue(maxsize=num_cached)
    sentinel = object()

    def producer():
        for item in iterator:
            queue.put(item)
        queue.put(sentinel)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not sentinel:
            yield item
            queue.task_done()
            item = queue.get()



