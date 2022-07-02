from typing import List, Dict, Tuple, Optional

import numpy as np
from nptyping import NDArray, Shape, Int, Float

from dl_scratch2.common.layers import Layer, MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size: int, hidden_size: int):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in : NDArray[Shape['*, *'], Float] = 0.01 * np.random.randn(V, H).astype('f')
        W_out : NDArray[Shape['*, *'], Float] = 0.01 * np.random.randn(H, V).astype('f')

        # レイヤの生成
        self.in_layer0 : MatMul = MatMul(W_in)
        self.in_layer1 : MatMul = MatMul(W_in)
        self.out_layer : MatMul = MatMul(W_out)
        self.loss_layer : SoftmaxWithLoss = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        layers : List[MatMul] = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params : List[NDArray] = []
        self.grads : List[NDArray] = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs : NDArray[Shape['*, *'], Float] = W_in

    def forward(self, contexts: NDArray[Shape['*, *, 2'], Int], target: NDArray[Shape['*, 2'], Int]) -> float:
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
