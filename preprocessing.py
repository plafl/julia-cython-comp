import numpy as np
import scipy.sparse


def load_lastfm(path="./lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"):
    def index(x, name):
        i = x.get(name)
        if i is None:
            x[name] = i = len(x)
        return i

    n_lines = 0
    with open(path, "r") as inp:
        for line in inp:
            n_lines += 1
    data = np.empty((n_lines, 3), dtype=int)
    users = {}
    items = {}
    with open(path, "r") as inp:
        for i, line in enumerate(inp):
            cells = line.split('\t')
            data[i, 0] = index(users, cells[0])
            data[i, 1] = index(items, cells[1])
            data[i, 2] = int(cells[3])
    return data


lastfm = load_lastfm()
lastfm = scipy.sparse.csr_matrix(
    (lastfm[:, 2], (lastfm[:, 0], lastfm[:, 1])), dtype=np.float32)
scipy.sparse.save_npz('lastfm.npz', lastfm)
