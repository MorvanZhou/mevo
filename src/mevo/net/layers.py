import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from mevo.net import activations


class Layer(ABC):
    def __init__(
            self,
            w_shape: tp.Sequence[int],
            activation: tp.Union[str, activations.Activation],
            w_initializer,
            b_initializer,
            use_bias: bool,
    ):
        self.w = np.empty(w_shape, dtype=np.float32)
        self.b = None
        self.use_bias = use_bias
        if self.use_bias:
            shape = [1] * len(w_shape)
            shape[-1] = w_shape[-1]  # only have bias on the last dimension
            self.b = np.empty(shape, dtype=np.float32)

        if activation is None:
            self.activation = activations.linear
        elif isinstance(activation, str):
            self.activation = activations.ACTIVATION_MAP[activation]
        else:
            self.activation = activation

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Dense(Layer):
    def __init__(
            self,
            in_units: int,
            out_units: int,
            activation: tp.Union[str, activations.Activation] = None,
            w_initializer=None,
            b_initializer=None,
            use_bias: bool = True,
    ):
        super().__init__(
            w_shape=(in_units, out_units),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias
        )

    def forward(self, x: tp.Union[np.ndarray]) -> np.ndarray:
        o = x.dot(self.w)
        if self.use_bias:
            o += self.b

        o = self.activation(o)
        return o


class Conv2D(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tp.Union[int, tp.Sequence[int]] = (3, 3),
            strides: tp.Union[int, tp.Sequence[int]] = (1, 1),
            padding: str = 'valid',
            channels_last: bool = True,
            activation: tp.Union[str, activations.Activation] = None,
            w_initializer=None,
            b_initializer=None,
            use_bias: bool = True,
    ):
        self.kernel_size = _get_tuple(kernel_size)
        self.strides = _get_tuple(strides)
        super().__init__(
            w_shape=(in_channels,) + self.kernel_size + (out_channels,),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError

        self.channels_last = channels_last

    def forward(self, x: tp.Union[np.ndarray]) -> np.ndarray:
        if not self.channels_last:
            # [batch, channel, height, width] => [batch, height, width, channel]
            x = np.transpose(x, (0, 2, 3, 1))

        padded, tmp_conved = _get_padded_and_tmp_out(
            x, self.kernel_size, self.strides, self.out_channels, self.padding
        )
        o = self.convolution(padded, self.w, tmp_conved)
        if self.use_bias:
            o += self.b
        o = self.activation(o)
        if not self.channels_last:
            o = o.transpose((0, 3, 1, 2))
        return o

    def convolution(self, x: np.ndarray, flt: np.ndarray, conved: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        t_flt = flt.transpose((1, 2, 0, 3))  # [c,h,w,out] => [h,w,c,out]
        s0, s1, k0, k1 = self.strides + tuple(flt.shape[1:3])
        for i in range(0, conved.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, conved.shape[2]):  # in each column of the convoluted feature map
                x_seg_matrix = x[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :].reshape(
                    (batch_size, -1))  # [n,h,w,c] => [n, h*w*c]
                flt_matrix = t_flt.reshape((-1, flt.shape[-1]))  # [h,w,c, out] => [h*w*c, out]
                filtered = x_seg_matrix.dot(flt_matrix)  # sum of filtered window [n, out]
                conved[:, i, j, :] = filtered
        return conved

    def fast_convolution(self, x: np.ndarray, flt: np.ndarray, conved: np.ndarray) -> np.ndarray:
        # according to:
        # http://fanding.xyz/2017/09/07/CNN%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%9A%84Python%E5%AE%9E%E7%8E%B0III-CNN%E5%AE%9E%E7%8E%B0/

        # create patch matrix
        oh, ow, sh, sw, fh, fw = [conved.shape[1], conved.shape[2], self.strides[0],
                                  self.strides[1], flt.shape[1], flt.shape[2]]
        n, h, w, c = x.shape
        shape = (n, oh, ow, fh, fw, c)
        strides = (c * h * w, sh * w, sw, w, 1, h * w)
        strides = x.itemsize * np.array(strides)
        x_col = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
        x_col = np.ascontiguousarray(x_col)  # padded [n,h,w,c] => [n*oh*ow, h*w*c]
        x_col.shape = (n * oh * ow, fh * fw * c)  # [n*oh*ow, fh*fw*c]
        w_t = flt.transpose((1, 2, 0, 3)).reshape(-1, self.out_channels)  # => [hwc, oc]

        # IMPORTANT! as_stride function has some wired behaviours
        # which gives a not accurate result (precision issue) when performing matrix dot product.
        # I have compared the fast convolution with normal convolution and cannot explain the precision issue.
        wx = x_col.dot(w_t)  # [n*oh*ow, fh*fw*c] dot [fh*fw*c, oc] => [n*oh*ow, oc]
        return wx.reshape(conved.shape)


class _Pool(Layer, ABC):
    def __init__(
            self,
            kernel_size: tp.Union[int, tp.Sequence[int]] = (3, 3),
            strides: tp.Union[int, tp.Sequence[int]] = (1, 1),
            padding: str = "valid",
            channels_last: bool = True,
    ):
        super().__init__()
        self.kernel_size = _get_tuple(kernel_size)
        self.strides = _get_tuple(strides)
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError
        self.channels_last = channels_last
        self._padded = None

    def forward(self, x):
        if not self.channels_last:  # "channels_first":
            # [batch, channel, height, width] => [batch, height, width, channel]
            x = np.transpose(x, (0, 2, 3, 1))
        padded, o = _get_padded_and_tmp_out(
            x, self.kernel_size, self.strides, x.shape[-1], self.padding)
        s0, s1, k0, k1 = self.strides + self.kernel_size
        for i in range(0, o.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, o.shape[2]):  # in each column of the convoluted feature map
                window = padded[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :]  # [n,h,w,c]
                o[:, i, j, :] = self.agg_func(window)
        if not self.channels_last:
            o = o.transpose((0, 3, 1, 2))
        return o

    @staticmethod
    @abstractmethod
    def agg_func(x):
        raise NotImplementedError


class MaxPool2D(_Pool):
    def __init__(
            self,
            pool_size: tp.Union[int, tp.Sequence[int]] = (3, 3),
            strides: tp.Union[int, tp.Sequence[int]] = (1, 1),
            padding: str = "valid",
            channels_last: bool = True,
    ):
        super().__init__(
            kernel_size=pool_size,
            strides=strides,
            padding=padding,
            channels_last=channels_last,
        )

    @staticmethod
    def agg_func(x):
        return x.max(axis=(1, 2))


class AvgPool2D(_Pool):
    def __init__(
            self,
            pool_size: tp.Union[int, tp.Sequence[int]] = (3, 3),
            strides: tp.Union[int, tp.Sequence[int]] = (1, 1),
            padding: str = "valid",
            channels_last: bool = True,
    ):
        super().__init__(
            kernel_size=pool_size,
            strides=strides,
            padding=padding,
            channels_last=channels_last, )

    @staticmethod
    def agg_func(x):
        return x.mean(axis=(1, 2))


class Flatten:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x.reshape((x.shape[0], -1))


def _get_tuple(inputs):
    if isinstance(inputs, (tuple, list)):
        out = tuple(inputs)
    elif isinstance(inputs, int):
        out = (inputs, inputs)
    else:
        raise TypeError
    return out


def _get_padded_and_tmp_out(img, kernel_size, strides, out_channels, padding):
    # according to: http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
    batch, h, w = img.shape[:3]
    (fh, fw), (sh, sw) = kernel_size, strides

    if padding == "same":
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))
        ph = int(np.max([0, (out_h - 1) * sh + fh - h]))
        pw = int(np.max([0, (out_w - 1) * sw + fw - w]))
        pt, pl = int(np.floor(ph / 2)), int(np.floor(pw / 2))
        pb, pr = ph - pt, pw - pl
    elif padding == "valid":
        out_h = int(np.ceil((h - fh + 1) / sh))
        out_w = int(np.ceil((w - fw + 1) / sw))
        pt, pb, pl, pr = 0, 0, 0, 0
    else:
        raise ValueError
    padded_img = np.pad(img, ((0, 0), (pt, pb), (pl, pr), (0, 0)), 'constant', constant_values=0.).astype(np.float32)
    tmp_out = np.zeros((batch, out_h, out_w, out_channels), dtype=np.float32)
    return padded_img, tmp_out
