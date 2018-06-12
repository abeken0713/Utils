import numpy as np
from pathlib import Path
import os

def checkPath(path):
    return path.exists()

def checkAndMakeDir(dirpath):
    flag = checkPath(dirpath)
    if not flag:
        dirpath.mkdir()
    return flag

def splitbatch(data, batch_size=None, n_batch=None, shuffle=False, seed=None):
    """
    [Arguments]
    - data: (tuple of) numpy.ndarray; shape should be identical
    - batch_size or n_batch: int; if both given, n_batch will be computed depending on batch_size
    - shuffle: boolean; if True, order of the first dimension will be shuffled (default False)
    - seed: int; random seed for shuffling
    """
    if (batch_size is None or type(batch_size) is not int) and (n_batch is None or type(n_batch) is not int):
        raise AttributeError("`batch_size` or `n_batch` should be int")
    if type(data) is np.ndarray:
        data = [data]
    if not all(map(lambda x: x.shape[0] == data[0].shape[0], data)):
        raise AttributeError("`data` should contain arrays with the same size of dim0")
    N = data[0].shape[0]
    if n_batch is None or (batch_size is not None and n_batch is not None):
        n_batch = N // batch_size + (N % batch_size != 0)
    if shuffle:
        rng = np.random.RandomState(seed)
        idx_list = rng.permutation(N)
    else:
        idx_list = np.arange(N)
    for i in range(n_batch):
        start = i*batch_size
        end = (i+1)*batch_size
        if end > N:
            end = None
        idx = idx_list[start:end]
        yield list(map(lambda d: d[idx], data))




