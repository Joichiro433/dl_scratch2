from typing import List, Optional
from abc import ABCMeta, abstractmethod

from nptyping import NDArray, Shape, Int, Float

from dl_scratch2.common.np import *  # import numpy as np
from dl_scratch2.common.config import GPU
from dl_scratch2.common.functions import softmax, cross_entropy_error


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self):
        raise NotImplementedError()


class MatMul(Layer):
    def __init__(self, W: NDArray) -> None:
        self.params : List[NDArray] = [W]
        self.grads : List[NDArray] = [np.zeros_like(W)]
        self.x : Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        W, = self.params
        out : NDArray = x @ W
        self.x = x
        return out

    def backward(self, dout: NDArray) -> NDArray:
        W, = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        self.grads[0][...] = dW
        return dx


class Affine(Layer):
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax(Layer):
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid(Layer):
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss(Layer):
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout(Layer):
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding(Layer):
    def __init__(self, W: NDArray[Shape['Vocab, Output'], Float]):
        self.params : List[NDArray] = [W]
        self.grads : List[NDArray] = [np.zeros_like(W)]
        self.idx : Optional[List[int]] = None

    def forward(self, idx: NDArray[Shape['Batch'], Int]) -> NDArray[Shape['Batch, Output'], Float]:
        W, = self.params
        self.idx = idx
        out : NDArray[Shape['Batch, Output'], Float] = W[idx]
        return out

    def backward(self, dout: NDArray[Shape['Batch, Output'], Float]) -> None:
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
