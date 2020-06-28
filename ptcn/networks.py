from typing import Tuple

import tensorflow as tf

from ptcn.components import TemporalResidualBlock, PERMITTED_PADDING_TYPES
from ptcn.constants import DEFAULT_RANDOM_SEED


class GenericTemporalConvNet(tf.keras.Model):
    problem_type_activations = {
        "generic": "relu",
        "regression": "linear",
        "classification": "sigmoid"
    }

    def __init__(
            self,
            n_filters: Tuple[int, ...],
            kernel_size: int = 3,
            padding_type: str = 'causal',
            dropout_rate: float = 0.2,
            problem_type: str = 'generic'
    ):
        """A generic Temporal Convolutional Network

        Expects data to be provided in the following shape: (batch_size, n_timesteps, n_features)

        :param n_filters:
        :param kernel_size:
        :param padding_type:
        :param dropout_rate:
        """
        super(GenericTemporalConvNet, self).__init__()
        # validate args
        assert padding_type in PERMITTED_PADDING_TYPES, f"Value of 'padding_type' parameter must be one of {PERMITTED_PADDING_TYPES}"
        assert 0. < dropout_rate < 1., r"Value of 'dropout_rate' parameter must lie inside the range (0., 1.)"
        # TODO: validate n_filters
        assert problem_type in ['generic', 'regression', 'classification']

        # Create network
        self.network = tf.keras.Sequential(
            [
                self._initialise_res_block(
                    dropout_rate,
                    kernel_size,
                    layer_filters,
                    layer_idx,
                    n_filters,
                    padding_type,
                    problem_type,
                )
                for layer_idx, layer_filters in enumerate(n_filters)
            ]
        )

    def _initialise_res_block(self, dropout_rate, kernel_size, layer_filters, layer_idx, n_filters, padding_type,
                              problem_type):
        layer_activation = self._get_activation(layer_idx, n_filters, problem_type)
        return TemporalResidualBlock(
            dilation=2 ** layer_idx,
            n_filters=layer_filters,
            kernel_size=kernel_size,
            padding_type=padding_type,
            dropout_rate=dropout_rate,
            final_activation=layer_activation,
            random_seed=DEFAULT_RANDOM_SEED * (layer_idx + 1),
            name=f"resblock_{layer_idx}"
        )

    def _get_activation(self, layer_idx: int, n_filters: Tuple, problem_type: str):
        return self.problem_type_activations.get(problem_type) if layer_idx + 1 == len(n_filters) else 'linear'

    def call(self, inputs, training=None, mask=None):
        return self.network(inputs, training=training)
