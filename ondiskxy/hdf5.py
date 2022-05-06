import os
import h5py
import json
import numpy as np
import shutil

from .utils import slicer


class Hdf5Writer(object):
    def __init__(self, file_name):
        if file_name is None:
            self.file_name = None
        else:
            self.file_name = os.path.abspath(file_name)
        self.name = "data"

    def set_file(self, file_name):
        self.file_name = os.path.abspath(file_name)

    def append(self, X):
        if not os.path.exists(self.file_name):
            maxshape = list(X.shape)
            maxshape[0] = None
            maxshape = tuple(maxshape)
            with h5py.File(self.file_name, "w") as f:
                f.create_dataset(
                    self.name,
                    shape=X.shape,
                    data=X,
                    maxshape=maxshape,
                    chunks=True,
                    dtype=X.dtype,
                )
        else:
            with h5py.File(self.file_name, "a") as f:
                f[self.name].resize((f[self.name].shape[0] + X.shape[0]), axis=0)
                f[self.name][-X.shape[0] :] = X


class Hdf5Reader(object):
    def __init__(self, file_name):
        self.file_name = os.path.abspath(file_name)
        assert os.path.exists(self.file_name)
        self.name = "data"

    def read(self):
        with h5py.File(self.file_name, "r") as f:
            X = f[self.name][:]
        return X

    def iterchunks(self, chunksize=1000):
        with h5py.File(self.file_name, "r") as f:
            n = f[self.name].shape[0]
            for i_start, i_end in slicer(n, chunksize):
                yield f[self.name][i_start:i_end]

    def iterrows(self):
        for chunk in self.iterchunks():
            for i in range(chunk.shape[0]):
                yield chunk[i]


class MultiHdf5Writer(object):
    def __init__(self, dir_path, max_file_size_mb=100, max_rows_per_file=1000000):
        self.dir_path = os.path.abspath(dir_path)
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.max_rows_per_file = max_rows_per_file
        self.rows_per_file = None
        self.meta_file = os.path.join(self.dir_path, "meta.json")
        self.writer = Hdf5Writer(None)
        self.cur_file_idx = 0
        self.cur_offset = 0

    def _get_rows_per_file(self, row):
        row_size = row.nbytes / 1048576  # bytes to megabytes
        nrows = self.max_file_size_mb / row_size
        nrows = int(np.clip(nrows, 1, self.max_rows_per_file))
        return nrows

    def _update_meta(self, file_idx, start_idx, end_idx):
        if not os.path.exists(self.meta_file):
            with open(self.meta_file, "w") as f:
                meta = [[file_idx, start_idx, end_idx]]
                json.dump(meta, f, indent=4)
        else:
            with open(self.meta_file, "r") as f:
                meta = json.load(f)
            if file_idx == int(meta[-1][0]):
                meta[-1][-1] = end_idx
            else:
                meta += [[file_idx, start_idx, end_idx]]
            with open(self.meta_file, "w") as f:
                json.dump(meta, f, indent=4)

    def _read_meta(self):
        with open(self.meta_file, "r") as f:
            meta = json.load(f)
        return meta

    def _get_file_name(self, file_idx):
        file_name = os.path.join(self.dir_path, "slice-{0}.h5".format(file_idx))
        return file_name

    def _count_current_rows(self, file_name):
        if not os.path.exists(file_name):
            return 0
        else:
            with h5py.File(file_name, "r") as f:
                return f[self.writer.name].shape[0]

    def append(self, X):
        if self.rows_per_file is None:
            self.rows_per_file = self._get_rows_per_file(X[0])
        for start_idx, end_idx in slicer(X.shape[0], self.rows_per_file):
            X_ = X[start_idx:end_idx]
            file_name = self._get_file_name(self.cur_file_idx)
            cur_rows = self._count_current_rows(file_name)
            this_rows = X_.shape[0]
            end_idx = start_idx + this_rows
            if cur_rows + this_rows > self.rows_per_file:
                self.cur_file_idx += 1
                file_name = self._get_file_name(self.cur_file_idx)
            start_idx += self.cur_offset
            end_idx += self.cur_offset
            self._update_meta(self.cur_file_idx, start_idx, end_idx)
            self.writer.set_file(file_name)
            self.writer.append(X_)
        self.cur_offset = end_idx


class MultiHdf5Reader(object):
    def __init__(self, dir_path):
        self.dir_path = os.path.abspath(dir_path)
        assert os.path.exists(self.dir_path)
        self.meta_file = os.path.join(self.dir_path, "meta.json")

    def _get_file_path(self, file_idx):
        return os.path.join(self.dir_path, "slice-{0}.h5".format(file_idx))

    def _read_meta(self):
        with open(self.meta_file, "r") as f:
            return json.load(f)

    def iterfiles(self):
        meta = self._read_meta()
        for r in meta:
            file_idx = r[0]
            yield self._get_file_path(file_idx)

    def read(self):
        X = None
        for file_name in self.iterfiles():
            reader = Hdf5Reader(file_name)
            X_ = reader.read()
            if X is None:
                X = X_
            else:
                X = np.vstack((X, X_))
        return X

    def iterrows(self):
        for file_name in self.iterfiles():
            reader = Hdf5Reader(file_name)
            for row in reader.iterrows():
                yield row
