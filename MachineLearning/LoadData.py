import gzip
import numpy as np
import pickle
from pathlib import Path
from sklearn.datasets import fetch_mldata
from .utils import splitbatch

def load_mnist(dirpath="./Data/mnist", seed=1234):
    mypath = Path(__file__).absolute()
    dirpath = mypath.parent / dirpath
    mnist = fetch_mldata("MNIST original", data_home=str(dirpath))
    data = mnist["data"]
    target = mnist["target"]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(data.shape[0])
    data = data[idx]
    target = target[idx]
    train_size = 60000
    return data[:train_size], data[train_size:], target[:train_size], target[train_size:]

def load_cifar10(dirpath="./Data/cifar10/cifar-10-batches-py"):
    '''
    [Attributes]
    dirpath: relative directory path for the data
    [Outputs]
    train_data: 50000 x 3072 (= 3 x 32 x 32)
    test_data: 10000 x 3072
    train_label: 50000
    test_label: 10000
    '''
    mypath = Path(__file__).absolute()
    dirpath = mypath.parent / dirpath
    def unpickle(file):
        with open(str(dirpath / file), "rb") as f:
            dic = pickle.load(f, encoding="bytes")
        return dic
    train_batches = [unpickle("data_batch_%d" % i) for i in range(1, 6)]
    train_data = np.vstack([batch[b"data"] for batch in train_batches])
    train_label = np.hstack([np.array(batch[b"labels"]) for batch in train_batches])

    test_batch = unpickle("test_batch")
    test_data = test_batch[b"data"]
    test_label = np.array(test_batch[b"labels"])

    meta_batch = unpickle("batches.meta")
    label_names = [s.decode() for s in meta_batch[b"label_names"]]

    return label_names, train_data, test_data, train_label, test_label

# def load_mnist2(dirpath="./Data/mnist"):
#     """
#     load mnist train-test sets from filepath
#     train:test = 
#     returns train_data, train_label, test_data, test_label
#     """
#     mypath = Path(__file__).absolute()
#     dirpath = mypath.parent / dirpath
    
#     image_size = 28*28
#     filename = [
#         "train-images-idx3-ubyte.gz",
#         "train-labels-idx1-ubyte.gz",
#         "t10k-images-idx3-ubyte.gz",
#         "t10k-labels-idx1-ubyte.gz"
#     ]

#     if not (checkAndMakeDir(dirpath) and all(map(lambda s: checkPath(dirpath / s), filename))):
#         import shutil
#         import requests
#         for s in filename:
#             URL = "http://yann.lecun.com/exdb/mnist/" + s
#             print("Downloading %s" % URL)
#             res = requests.get(URL, stream=True)
#             with (dirpath / s).open("wb") as f:
#                 shutil.copyfileobj(res.raw, f)

#     out = []
    
#     for s in filename:
#         if "images" in s:
#             offset = 16
#         else:
#             offset = 8
#         with gzip.open(str(dirpath / s), "rb") as f:
#             data = np.frombuffer(f.read(), np.uint8, offset=offset)
#         if "images" in s:
#             data = data.reshape(-1, image_size)
#         out.append(data)

#     return out

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     train_data, train_label, test_data, test_label = load_mnist()
#     n_row, n_col = 9, 9
#     fig = plt.figure()
#     for i in range(n_row):
#         for j in range(n_col):
#             n = i*n_col + j
#             img = train_data[n].reshape(28, 28)
#             lbl = train_label[n+8]
#             ax = fig.add_subplot(n_row, n_col, n + 1)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_title(lbl)
#             ax.imshow(img, "Greys")
#     plt.show()