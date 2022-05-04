import os
import h5py
from .utils import slicer


class Hdf5XWriter(object):

    def __init__(self, file_name):
        self.file_name = os.path.abspath(file_name)
        self.name = "X"
        print("HDF5 file: {0}".format(self.file_name))
    
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
                    dtype=X.dtype)
        else:
            with h5py.File(self.file_name, "a") as f:
                f[self.name].resize((f[self.name].shape[0] + X.shape[0]), axis=0)
                f[self.name][-X.shape[0]:] = X


class Hdf5XReader(object):

    def __init__(self, file_name):
        self.file_name = os.path.abspath(file_name)
        assert os.path.exists(self.file_name)
        self.name = "X"
        self.chunksize = 1000
        print("HDF5 file: {0}".format(self.file_name))
    
    def read(self):
        with h5py.File(self.file_name, "r") as f:
            X = f["X"][:]
        return X

    def iterchunks(self):
        with h5py.File(self.file_name, "r") as f:
            n = f[self.name].shape[0]
            for i_start, i_end in slicer(n, self.chunksize):
                yield f[self.name][i_start:i_end]

    def iterrows(self):
        for chunk in self.iterchunks():
            for i in range(chunk.shape[0]):
                yield chunk[i]