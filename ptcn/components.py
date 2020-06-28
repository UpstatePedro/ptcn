import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from ptcn.constants import DEFAULT_RANDOM_SEED

PERMITTED_PADDING_TYPES = ['valid', 'causal', 'same']


class TemporalResidualBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            dilation: int,
            n_filters: int,
            kernel_size: int = 3,
            padding_type: str = 'causal',
            dropout_rate: float = 0.2,
            final_activation: str = 'relu',
            name: str = 'TemporalResBlock',
            random_seed: int = DEFAULT_RANDOM_SEED
    ):
        """Residual Block for TemporalConvNet

        :param dilation:
        :param n_filters:
        :param kernel_size:
        :param padding_type:
        :param dropout_rate:
        :param random_seed:
        """
        super(TemporalResidualBlock, self).__init__(name=name)

        # Set the random seed
        self.random_seed = random_seed

        # Validate parameters
        assert dilation in np.logspace(0, 20, num=21, base=2), "Value of 'dilation' parameter must be a power of 2"
        assert padding_type in PERMITTED_PADDING_TYPES, f"Value of 'padding_type' parameter must be one of {PERMITTED_PADDING_TYPES}"
        assert 0. < dropout_rate < 1., r"Value of 'dropout_rate' parameter must lie inside the range (0., 1.)"
        assert final_activation in ['relu', 'linear', 'sigmoid']

        # Gaussian parameter initialisations
        initialiser = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=self.random_seed)

        # First block
        self.conv_1 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding_type,
            dilation_rate=dilation,
            activation='tanh',
            kernel_initializer=initialiser
        )
        self.norm_conv_1 = tfa.layers.WeightNormalization(self.conv_1)
        self.activation_1 = layers.Activation('tanh')
        self.dropout_1 = layers.Dropout(rate=dropout_rate, seed=self.random_seed)

        # Second block
        self.conv_2 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding_type,
            dilation_rate=dilation,
            activation='tanh',
            kernel_initializer=initialiser
        )
        self.norm_conv_2 = tfa.layers.WeightNormalization(self.conv_2)
        self.activation_2 = layers.Activation('tanh')
        self.dropout_2 = layers.Dropout(rate=dropout_rate, seed=self.random_seed)

        # Correct sizing
        self.shape_regulator = layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            dilation_rate=1,
            activation='linear',
            kernel_initializer=initialiser
        )

        # End-of-block activation (incorporates the skip connection)
        self.final_activation = layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        initial_inputs = inputs
        x = self.norm_conv_1(inputs)
        x = self.activation_1(x)
        x = self.dropout_1(x) if training else x

        x = self.norm_conv_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x) if training else x

        if initial_inputs.shape[-1] != x.shape[-1]:
            initial_inputs = self.shape_regulator(initial_inputs)
        assert initial_inputs.shape[-1] == x.shape[-1], f"initial shape {initial_inputs.shape} does not match x shape {x.shape}"

        return self.final_activation(initial_inputs + x)
