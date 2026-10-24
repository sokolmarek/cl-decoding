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
"""Multimodal Efficient-CapsNet for cognitive load classification (Section 4.6, Fig. 5)."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from .layers import FCCaps, Length, Mask, PrimaryCaps

CAPSNET_NAME = "Efficient_CapsNet"


def _conv_block(x, filters, kernel_size, strides, name):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="valid", activation="relu",
                               kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), name=name)(x)
    return tf.keras.layers.BatchNormalization()(x)


def capsnet(input_shape, n_classes=2):
    """Encoder, primary capsules and class capsules.

    Returns a model with outputs ``(class_capsules, class_scores)``.
    """
    x = tf.keras.layers.Input(input_shape)

    h = _conv_block(x, 32, 2, 1, "conv1")
    h = _conv_block(h, 64, 5, 1, "conv2")
    h = tf.keras.layers.MaxPool2D(pool_size=2)(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = _conv_block(h, 64, 3, 1, "conv3")
    h = _conv_block(h, 128, 3, 2, "conv4")
    h = tf.keras.layers.MaxPool2D(pool_size=2)(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    primary = PrimaryCaps(F=128, K=9, D=8)(h)
    class_caps = FCCaps(n_classes, 16)(primary)
    scores = Length(name="length_capsnet_output")(class_caps)
    return tf.keras.Model(inputs=x, outputs=[class_caps, scores], name=CAPSNET_NAME)


def decoder(input_shape, n_classes=2):
    """Fully connected decoder that reconstructs the input from the masked class capsules."""
    inputs = tf.keras.Input(16 * n_classes)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(int(np.prod(input_shape)), activation="sigmoid", kernel_initializer="glorot_normal")(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name="out_generator")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="Generator")


def build_model(input_shape, n_classes=2):
    """Training model (masked by true labels) and evaluation model (masked by prediction).

    The training model takes ``[x, y_one_hot]`` and returns ``[class_scores, reconstruction]``;
    the evaluation model takes ``x`` and returns the same outputs.
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.Input(shape=(n_classes,))

    caps = capsnet(input_shape, n_classes)
    gen = decoder(input_shape, n_classes)
    class_caps, scores = caps(inputs)

    train_model = tf.keras.Model([inputs, y_true], [scores, gen(Mask()([class_caps, y_true]))],
                                 name="Efficient_CapsNet_Generator")
    eval_model = tf.keras.Model(inputs, [scores, gen(Mask()(class_caps))], name="Efficient_CapsNet_Generator")
    return train_model, eval_model


def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lbd=0.5):
    """Capsule margin loss."""
    L = y_true * tf.square(tf.maximum(0.0, m_plus - y_pred)) + \
        lbd * (1 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))
