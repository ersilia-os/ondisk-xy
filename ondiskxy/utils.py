def chunker(seq, size):
    for x in range(0, len(seq), size):
        yield seq[x : x + size]


def slicer(n, size):
    for x in range(0, n, size):
        yield (x, x + size)



def train_test_split(y, test_size):
    train_idxs = ""
    test_idxs = ""
    return train_idxs, test_idxs