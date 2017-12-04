# coding=utf-8

import mxnet as mx
from mxnet import nd


def try_gpu():
    """
    If GPU is available, return all available GPUs, else return mx.cpu()
    :return:
    """
    ctx_lsit = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx)
            ctx_lsit.append(ctx)
    except:
        pass

    if not ctx_lsit:
        ctx_lsit = [mx.cpu()]
    return ctx_lsit


# TODO: to complete
class DataLoader(object):
    """
    Similar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more examples each time.
    But the limits are: 1) all examples in dataset have the same shape; 2) data
    transformer needs to process multiple examples at each time.
    """
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = data[1]


if __name__ == '__main__':
    print(try_gpu())
