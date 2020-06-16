import pytest
import numpy as np
import tensorflow as tf

from ptcn.components import TemporalResidualBlock


def test_initialise_TemporalResidualBlock_is_keras_model():
    residual_block = TemporalResidualBlock(dilation=2, n_filters=5)
    assert isinstance(residual_block, tf.keras.Model)


def test_initialise_TemporalResidualBlock_invalid_dilation():
    with pytest.raises(AssertionError, match=r"Value of 'dilation' parameter"):
        residual_block = TemporalResidualBlock(dilation=3.5, n_filters=5)


def test_initialise_TemporalResidualBlock_invalid_padding_type():
    with pytest.raises(AssertionError, match=r"Value of 'padding_type' parameter"):
        residual_block = TemporalResidualBlock(dilation=2, n_filters=5, padding_type='bla')


def test_initialise_TemporalResidualBlock_invalid_dropout_rate():
    with pytest.raises(AssertionError, match=r"Value of 'dropout_rate' parameter"):
        residual_block = TemporalResidualBlock(dilation=2, n_filters=5, dropout_rate=40)


def test_call_with_array():
    batch_size = 1
    sequence_len = 5
    input_variables = 1
    n_filters = 5

    input_arr = np.random.rand(batch_size, sequence_len, input_variables)
    residual_block = TemporalResidualBlock(dilation=2, n_filters=n_filters)
    exp_output_shape = [batch_size, sequence_len, n_filters]
    output = residual_block(input_arr)
    np.testing.assert_equal(output.shape, exp_output_shape)