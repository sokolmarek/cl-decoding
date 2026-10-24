# Adapted from Efficient-CapsNet, Copyright 2021 Vittorio Mazzia & Francesco Salvetti.
# https://github.com/EscVM/Efficient-CapsNet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capsule layers with self-attention routing."""

import tensorflow as tf


class Squash(tf.keras.layers.Layer):
    """Squash activation of Efficient-CapsNet: ``(1 - 1 / e^||s||) * s / ||s||``."""

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s, axis=-1, keepdims=True)
        return (1 - 1 / (tf.math.exp(n) + self.eps)) * (s / (n + self.eps))

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}


class PrimaryCaps(tf.keras.layers.Layer):
    """Primary capsules from a depthwise convolution.

    Parameters
    ----------
    F : int
        Number of depthwise convolution filters.
    K : int
        Kernel size.
    D : int
        Capsule dimension.
    s : int
        Strides.
    """

    def __init__(self, F, K, D, s=1, **kwargs):
        super().__init__(**kwargs)
        self.F, self.K, self.D, self.s = F, K, D, s

    def build(self, input_shape):
        self.dw_conv = tf.keras.layers.Conv2D(self.F, self.K, self.s, activation="linear", groups=self.F,
                                              padding="same")
        self.dw_conv.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.dw_conv(inputs)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3] // self.D, self.D])
        return Squash()(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2] * self.F // self.D, self.D)

    def get_config(self):
        return {**super().get_config(), "F": self.F, "K": self.K, "D": self.D, "s": self.s}


class FCCaps(tf.keras.layers.Layer):
    """Fully connected capsule layer with self-attention routing.

    Parameters
    ----------
    N : int
        Number of output capsules.
    D : int
        Output capsule dimension.
    """

    def __init__(self, N, D, kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.N, self.D = N, D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        input_N, input_D = input_shape[-2], input_shape[-1]
        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D], initializer=self.kernel_initializer,
                                 name="W")
        self.b = self.add_weight(shape=[self.N, input_N, 1], initializer=tf.zeros_initializer(), name="b")
        self.built = True

    def call(self, inputs):
        u = tf.einsum("...ji,kjiz->...kjz", inputs, self.W)  # (batch, N, input_N, D)
        c = tf.einsum("...ij,...kj->...i", u, u)[..., None]  # (batch, N, input_N, 1)
        c = c / tf.sqrt(tf.cast(self.D, tf.float32))
        c = tf.nn.softmax(c, axis=1)
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c), axis=-2)  # (batch, N, D)
        return Squash()(s)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.N, self.D)

    def get_config(self):
        return {**super().get_config(), "N": self.N, "D": self.D}


class Length(tf.keras.layers.Layer):
    """Length of each capsule, used as class score."""

    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(tf.keras.layers.Layer):
    """Keep only one class capsule (true class if given, otherwise the longest) and flatten."""

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            capsules, mask = inputs
        else:
            capsules = inputs
            lengths = tf.sqrt(tf.reduce_sum(tf.square(capsules), -1))
            mask = tf.one_hot(tf.argmax(lengths, 1), depth=capsules.shape[1])
        masked = capsules * tf.expand_dims(mask, -1)
        return tf.reshape(masked, [tf.shape(masked)[0], capsules.shape[1] * capsules.shape[2]])
