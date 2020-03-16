# coding: utf-8

import numpy as np
import pandas as pd
# from cached_property import cached_property

class Batch:

    def __init__(self, size):
        self.size = size
        self.keys = []

    def set(self, key, val):
        val = xt.as_ndarray(val)
        if hasattr(self, key):
            stored = self.__getattribute__(key)
            stored = np.concatenate([stored, val], axis=0)
            self.__setattr__(key, stored)
            return
        # new keyword
        self.keys.append(key)
        self.__setattr__(key, val)


class DataLoader:

    def __init__(
            self,
            batch_size,
            at_random=True,
            files=None
    ):
        self._batch_size = batch_size
        self._at_random = at_random
        self.xs = np.array([])
        self.ys = np.array([])
        self._current = 0
        self._indices = []

    def __len__(self):
        return np.ceil(self.size() / self._batch_size).astype(int)

    def __iter__(self):
        self._current = 0
        self._indices = np.arange(self.size())
        if self._at_random:
            np.random.shuffle(self._indices)
        return self

    def __next__(self):
        if self._current >= self.size():
            raise StopIteration
        idx = self._indices[self._current:self._current+self._batch_size]
        self._current += self._batch_size
        batch = Batch(self._batch_size)
        batch.set("xs", self.xs[idx, :])
        batch.set("ys", self.ys[idx, :])
        return batch

    def size(self):
        return self.xs.shape[0]

    @staticmethod
    def load_single_file(file_name, pre_process=None):
        d = pd.read_csv(file_name)
        if pre_process is None:
            d = d.to_numpy()
            xs = d[:-1, :]
            ys = d[-1:, :]
        else:
            xs, ys = pre_process(d)
        return xs, ys

    def load_files(self, files, pre_process=None):
        xs = []
        ys = []
        for file in files:
            xs_, ys_ = self.load_single_file(file, pre_process)
            xs.append(xs_)
            ys.append(ys_)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        if len(self.xs) == 0:
            self.xs = xs
            self.ys = ys
        else:
            self.xs = np.concatenate([self.xs, xs])
            self.ys = np.concatenate([self.ys, ys])

        print("Data size::", "xs: ", self.xs.shape, "\t ys: ", ys.shape)

        return xs, ys


def stack_10step(data):
    size = 10
    source_xs = data[["dec", "u", "w", "q", "theta"]].to_numpy()
    source_ys = data[["mode"]].to_numpy()

    def retrieve_xs(k):
        return source_xs[k:size+k, :].flatten()

    def retrieve_ys(k):
        return source_ys[size+k, :].flatten()

    xs = np.array([retrieve_xs(k) for k in range(len(source_xs) - size)])
    ys = np.array([retrieve_ys(k) for k in range(len(source_ys) - size)])

    return xs, ys


if __name__ == '__main__':
    import xtools as xt
    xt.go_to_root()

    alls = [
        "result/dataset/normal/2020.03.15.181557/all.txt",
        "result/dataset/gain/2020.03.15.181538/all.txt",
        "result/dataset/rate/2020.03.15.181349/all.txt",
        "result/dataset/saturation/2020.03.15.181125/all.txt"
    ]

    loader = DataLoader(32)

    for all in alls:
        all = np.loadtxt(all, dtype=str)
        loader.load_files(all, pre_process=stack_10step)

    for batch in loader:
        print(batch.xs.shape)
