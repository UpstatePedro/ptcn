from typing import Tuple

import tensorflow as tf

from ptcn.components import TemporalResidualBlock, PERMITTED_PADDING_TYPES
from ptcn.constants import DEFAULT_RANDOM_SEED


class TemporalConvNet(tf.keras.Model):

    def __init__(
            self,
            n_filters: Tuple,
            kernel_size: int = 3,
            padding_type: str = 'causal',
            dropout_rate: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        # validate args
        assert padding_type in PERMITTED_PADDING_TYPES, f"Value of 'padding_type' parameter must be one of {PERMITTED_PADDING_TYPES}"
        assert 0. < dropout_rate < 1., r"Value of 'dropout_rate' parameter must lie inside the range (0., 1.)"
        # TODO: validate n_filters

        # Build network
        model = tf.keras.Sequential()
        for layer_idx, n_filters in enumerate(n_filters):
            residual_block = TemporalResidualBlock(
                dilation=2 ** layer_idx,
                n_filters=n_filters,
                kernel_size=kernel_size,
                padding_type=padding_type,
                dropout_rate=dropout_rate,
                random_seed=DEFAULT_RANDOM_SEED * (layer_idx + 1)
            )
            model.add(residual_block)
        self.network = model

    def call(self, inputs, training=None, mask=None):
        return self.network(inputs, training=training)
