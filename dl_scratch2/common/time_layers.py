from re import L
from typing import List, Dict, Tuple, Optional

from nptyping import NDArray, Shape, Int, Float

from dl_scratch2.common.np import *  # import numpy as np (or import cupy as np)
from dl_scratch2.common.layers import *
from dl_scratch2.common.functions import softmax, sigmoid


class RNN(Layer):
    def __init__(
            self, 
            Wx: NDArray[Shape['Input, Hidden'], Float], 
            Wh: NDArray[Shape['Hidden, Hidden'], Float], 
            b: NDArray[Shape['Hidden'], Float]) -> None:
        self.params : List[NDArray] = [Wx, Wh, b]
        self.grads : List[NDArray] = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache : Optional[Tuple[NDArray, NDArray, NDArray]] = None  # 逆伝播の計算に使用

    def forward(
            self, 
            x: NDArray[Shape['Batch, Input'], Float], 
            h_prev: NDArray[Shape['Batch, Hidden'], Float]) -> NDArray[Shape['Batch, Hidden'], Float]:
        Wx, Wh, b = self.params
        t : NDArray[Shape['Batch, Hidden'], Float] = (h_prev @ Wh) + (x @ Wx) + b
        h_next : NDArray[Shape['Batch, Hidden'], Float] = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(
            self, 
            dh_next: NDArray[Shape['Batch, Hidden'], Float]) -> Tuple[NDArray[Shape['Batch, Hidden'], Float], NDArray[Shape['Batch, Hidden'], Float]]:
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt : NDArray[Shape['Batch, Hidden'], Float] = dh_next * (1 - h_next ** 2)
        db : NDArray[Shape['Hidden'], Float] = np.sum(dt, axis=0)
        dWh : NDArray[Shape['Hidden, Hidden'], Float] = h_prev.T @ dt
        dh_prev : NDArray[Shape['Batch, Hidden'], Float] = dt @ Wh.T
        dWx : NDArray[Shape['Input, Hidden'], Float] = x.T @ dt
        dx : NDArray[Shape['Batch, Input'], Float] = dt @ Wx.T

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN(Layer):
    def __init__(
            self, 
            Wx : NDArray[Shape['Input, Hidden'], Float], 
            Wh : NDArray[Shape['Hidden, Hidden'], Float], 
            b : NDArray[Shape['Hidden'], Float], 
            stateful : bool = False) -> None:
        self.params : List[NDArray] = [Wx, Wh, b]
        self.grads : List[NDArray] = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers : Optional[List[RNN]] = None

        self.h : Optional[NDArray[Shape['Batch, Hidden'], Float]] = None  # ブロック間での h の引き継ぎ
        self.dh : Optional[NDArray[Shape['Batch, Hidden'], Float]] = None
        self.stateful : bool = stateful

    def forward(
            self, 
            xs : NDArray[Shape['Batch, TimeRange, Input'], Float]) -> NDArray[Shape['Batch, TimeRange, Hidden'], Float]:
        """forward関数

        Parameters
        ----------
        xs: NDArray[Shape[N, T, D], Float]
            N: バッチサイズ, T: 時系列データ数, D: 入力ベクトル次元
            
        Returns
        -------
        
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs : NDArray[Shape['Batch, TimeRange, Hidden'], Float] = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer : RNN = RNN(*self.params)
            self.h = layer.forward(x=xs[:, t, :], h_prev=self.h)  # 最後の h を保持する
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(
            self, 
            dhs: NDArray[Shape['Batch, TimeRange, Hidden'], Float]) -> NDArray[Shape['Batch, TimeRange, Input'], Float]:
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs : NDArray[Shape['Batch, TimeRange, Input'], Float] = np.empty((N, T, D), dtype='f')
        dh : NDArray[Shape['Batch, Hidden'], Float] = 0
        grads : List[NDArray] = [0, 0, 0]
        for t in reversed(range(T)):
            layer : RNN = self.layers[t]
            dx, dh = layer.backward(dh_next=dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class LSTM(Layer):
    def __init__(
            self, 
            Wx: NDArray[Shape['Input, Hidden_x_4'], Float], 
            Wh: NDArray[Shape['Hidden, Hidden_x_4'], Float], 
            b: NDArray[Shape['Hidden_x_4'], Float]) -> None:
        """
        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
        b: バイアス（4つ分のバイアスをまとめる）
        """
        self.params : List[NDArray] = [Wx, Wh, b]
        self.grads : List[NDArray] = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache : Optional[Tuple[NDArray, ...]] = None

    def forward(
            self, 
            x: NDArray[Shape['Batch, Input'], Float], 
            h_prev: NDArray[Shape['Batch, Hidden'], Float], 
            c_prev: NDArray[Shape['Batch, Hidden'], Float]
        ) -> Tuple[NDArray[Shape['Batch, Hidden'], Float], NDArray[Shape['Batch, Hidden'], Float]]:
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A : NDArray[Shape['Batch, Hidden_x_4'], Float] = (x @ Wx) + (h_prev @ Wh) + b

        f : NDArray[Shape['Batch, Hidden'], Float] = A[:, :H]
        g : NDArray[Shape['Batch, Hidden'], Float] = A[:, H:2*H]
        i : NDArray[Shape['Batch, Hidden'], Float] = A[:, 2*H:3*H]
        o : NDArray[Shape['Batch, Hidden'], Float] = A[:, 3*H:]

        f : NDArray[Shape['Batch, Hidden'], Float] = sigmoid(f)
        g : NDArray[Shape['Batch, Hidden'], Float] = np.tanh(g)
        i : NDArray[Shape['Batch, Hidden'], Float] = sigmoid(i)
        o : NDArray[Shape['Batch, Hidden'], Float] = sigmoid(o)

        c_next : NDArray[Shape['Batch, Hidden'], Float] = f * c_prev + g * i
        h_next : NDArray[Shape['Batch, Hidden'], Float] = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(
            self, 
            dh_next: NDArray[Shape['Batch, Hidden'], Float], 
            dc_next: NDArray[Shape['Batch, Hidden'], Float]
        ) -> Tuple[NDArray[Shape['Batch, Input'], Float], NDArray[Shape['Batch, Hidden'], Float], NDArray[Shape['Batch, Hidden'], Float]]:
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next : NDArray[Shape['Batch, Hidden'], Float] = np.tanh(c_next)

        ds : NDArray[Shape['Batch, Hidden'], Float] = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev : NDArray[Shape['Batch, Hidden'], Float] = ds * f

        di : NDArray[Shape['Batch, Hidden'], Float] = ds * g
        df : NDArray[Shape['Batch, Hidden'], Float] = ds * c_prev
        do : NDArray[Shape['Batch, Hidden'], Float] = dh_next * tanh_c_next
        dg : NDArray[Shape['Batch, Hidden'], Float] = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA : NDArray[Shape['Batch, Hidden_x_4'], Float] = np.hstack((df, dg, di, do))

        dWh : NDArray[Shape['Hidden, Hidden_x_4'], Float] = h_prev.T @ dA
        dWx : NDArray[Shape['Input, Hidden_x_4'], Float] = x.T @ dA
        db : NDArray[Shape['Hidden_x_4'], Float] = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx : NDArray[Shape['Batch, Input'], Float] = dA @ Wx.T
        dh_prev : NDArray[Shape['Batch, Hidden'], Float] = dA @ Wh.T

        return dx, dh_prev, dc_prev


class TimeLSTM(Layer):
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs : NDArray[Shape['*, *, *'], Float] = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads : List[NDArray] = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeEmbedding(Layer):
    def __init__(
            self, 
            W: NDArray[Shape['Vocab, Output'], Float]) -> None:
        self.params : List[NDArray] = [W]
        self.grads : List[NDArray] = [np.zeros_like(W)]
        self.layers : Optional[Embedding] = None
        self.W : NDArray[Shape['Vocab, Output'], Float] = W

    def forward(
            self, 
            xs: NDArray[Shape['Batch, TimeRange'], Float]) -> NDArray[Shape['Batch, TimeRange, Output'], Float]:
        N, T = xs.shape
        V, D = self.W.shape

        out : NDArray[Shape['Batch, TimeRange, Output'], Float] = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer : Embedding = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine(Layer):
    def __init__(
            self, 
            W: NDArray[Shape['Input, Output'], Float], 
            b: NDArray[Shape['Output'], Float]) -> None:
        self.params : List[NDArray] = [W, b]
        self.grads : List[NDArray] = [np.zeros_like(W), np.zeros_like(b)]
        self.x : Optional[NDArray[Shape['Batch, TimeRange, Input']]] = None

    def forward(
            self, 
            x: NDArray[Shape['Batch, TimeRange, Input'], Float]) -> NDArray[Shape['Batch, TimeRange, Output'], Float]:
        N, T, D = x.shape
        W, b = self.params

        rx : NDArray[Shape['Batch_x_TimeRange, Input'], Float] = x.reshape(N*T, -1)
        out : NDArray[Shape['Batch_x_TimeRange, Output'], Float] = rx @ W + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(
            self, 
            dout: NDArray[Shape['Batch, TimeRange, Output'], Float]) -> NDArray[Shape['Batch, TimeRange, Input'], Float]:
        x : NDArray[Shape['Batch, TimeRange, Input'], Float] = self.x
        N, T, D = x.shape
        W, b = self.params

        dout : NDArray[Shape['Batch_x_TimeRange, Output'], Float] = dout.reshape(N*T, -1)
        rx : NDArray[Shape['Batch_x_TimeRange, Input'], Float] = x.reshape(N*T, -1)

        db : NDArray[Shape['Output'], Float] = np.sum(dout, axis=0)
        dW : NDArray[Shape['Input, Output'], Float] = rx.T @ dout
        dx : NDArray[Shape['Batch_x_TimeRange, Input'], Float] = dout @ W.T
        dx : NDArray[Shape['Batch, TimeRange, Input'], Float] = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class TimeBiLSTM(Layer):
    def __init__(self, Wx1, Wh1, b1,
                 Wx2, Wh2, b2, stateful=False):
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs):
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out

    def backward(self, dhs):
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs

# ====================================================================== #
# 以下に示すレイヤは、本書で説明をおこなっていないレイヤの実装もしくは
# 処理速度よりも分かりやすさを優先したレイヤの実装です。
#
# TimeSigmoidWithLoss: 時系列データのためのシグモイド損失レイヤ
# GRU: GRUレイヤ
# TimeGRU: 時系列データのためのGRUレイヤ
# BiTimeLSTM: 双方向LSTMレイヤ
# Simple_TimeSoftmaxWithLoss：単純なTimeSoftmaxWithLossレイヤの実装
# Simple_TimeAffine: 単純なTimeAffineレイヤの実装
# ====================================================================== #


class TimeSigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


class GRU(Layer):
    def __init__(
            self, 
            Wx: NDArray[Shape['Input, Hidden_x_3'], Float], 
            Wh: NDArray[Shape['Hidden, Hidden_x_3'], Float], 
            b: NDArray[Shape['Hidden_x_3'], Float]) -> None:
        """

        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（3つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（3つ分の重みをまとめる）
        b: バイアス（3つ分のバイアスをまとめる）
        """
        self.params : List[NDArray] = [Wx, Wh, b]
        self.grads : List[NDArray] = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache : Optional[Tuple[NDArray, ...]] = None

    def forward(
            self, 
            x: NDArray[Shape['Batch, Input'], Float], 
            h_prev: NDArray[Shape['Batch, Hidden'], Float]
        ) -> NDArray[Shape['Batch, Hidden'], Float]:
        Wx, Wh, b = self.params
        H : int = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        z : NDArray[Shape['Batch, Hidden'], Float] = sigmoid((x @ Wxz) + (h_prev @ Whz) + bz)
        r : NDArray[Shape['Batch, Hidden'], Float] = sigmoid((x @ Wxr) + (h_prev @ Whr) + br)
        h_hat : NDArray[Shape['Batch, Hidden'], Float] = np.tanh((x @ Wxh) + (r*h_prev @ Whh) + bh)
        h_next : NDArray[Shape['Batch, Hidden'], Float] = (1-z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(
            self, 
            dh_next: NDArray[Shape['Batch, Hidden'], Float]
        ) -> Tuple[NDArray[Shape['Batch, Input'], Float], NDArray[Shape['Batch, Hidden'], Float]]:
        Wx, Wh, b = self.params
        H : int = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat : NDArray[Shape['Batch, Hidden'], Float] = dh_next * z
        dh_prev : NDArray[Shape['Batch, Hidden'], Float] = dh_next * (1-z)

        # tanh
        dt : NDArray[Shape['Batch, Hidden'], Float] = dh_hat * (1 - h_hat ** 2)
        dbh : NDArray[Shape['Hidden'], Float] = np.sum(dt, axis=0)
        dWhh : NDArray[Shape['Hidden, Hidden'], Float] = (r * h_prev).T @ dt
        dhr : NDArray[Shape['Batch, Hidden'], Float] = dt @ Whh.T
        dWxh : NDArray[Shape['Input, Hidden'], Float] = x.T @ dt
        dx : NDArray[Shape['Batch, Input'], Float] = dt @ Wxh.T
        dh_prev += r * dhr

        # update gate(z)
        dz : NDArray[Shape['Batch, Hidden'], Float] = dh_next * h_hat - dh_next * h_prev
        dt : NDArray[Shape['Batch, Hidden'], Float] = dz * z * (1-z)
        dbz : NDArray[Shape['Hidden'], Float] = np.sum(dt, axis=0)
        dWhz : NDArray[Shape['Hidden, Hidden'], Float] = h_prev.T @ dt
        dh_prev += dt @ Whz.T
        dWxz : NDArray[Shape['Input, Hidden'], Float] = x.T @ dt
        dx += dt @ Wxz.T

        # rest gate(r)
        dr : NDArray[Shape['Batch, Hidden'], Float] = dhr * h_prev
        dt : NDArray[Shape['Batch, Hidden'], Float] = dr * r * (1-r)
        dbr : NDArray[Shape['Hidden'], Float] = np.sum(dt, axis=0)
        dWhr : NDArray[Shape['Hidden, Hidden'], Float] = h_prev.T @ dt
        dh_prev += dt @ Whr.T
        dWxr : NDArray[Shape['Input, Hidden'], Float] = x.T @ dt
        dx += dt @ Wxr.T

        self.dWx : NDArray[Shape['Input, Hidden_x_3'], Float] = np.hstack((dWxz, dWxr, dWxh))
        self.dWh : NDArray[Shape['Hidden, Hidden_x_3'], Float] = np.hstack((dWhz, dWhr, dWhh))
        self.db : NDArray[Shape['Hidden_x_3'], Float] = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev


class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Simple_TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss

    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class Simple_TimeAffine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype='f')
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype='f')
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs
