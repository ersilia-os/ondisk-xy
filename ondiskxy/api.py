from .hdf5 import MultiHdf5Reader, MultiHdf5Writer, Hdf5Filter

class MatrixWriter(object):
    def __init__(self, dir_path, max_file_size_mb=100, max_rows_per_file=1000000):
        self.writer = MultiHdf5Writer(
            dir_path=dir_path,
            max_file_size_mb=max_file_size_mb,
            max_rows_per_file=max_rows_per_file,
        )

    def append(self, X):
        self.writer.append(X)


class MatrixReader(object):
    def __init__(self, dir_path):
        self.reader = MultiHdf5Reader(dir_path=dir_path)

    def read(self):
        return self.reader.read()

    def iterrows(self):
        for row in self.reader.iterrows():
            yield row

            
def filter_by_index(idxs, src_file, trg_file):
    d = Hdf5Filter(src_file)
    d.filter(idxs, trg_file)