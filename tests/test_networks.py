from ptcn.networks import GenericTemporalConvNet
import tensorflow as tf
import numpy as np


# def test_generictemporalconvnet():
#     tcn = GenericTemporalConvNet(
#         n_filters=(5, 5, 1)
#     )
#     tcn.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
#     # tcn.build(input_shape=(1, 5, 1))
#
#     x = tf.convert_to_tensor(np.random.random((100, 20, 2)))  # batch, seq_len, dim
#     y = tcn(x)
#
#     print(tcn.summary())
#     print(tcn.layers)


def test_generic_call_deterministic():
    x = tf.convert_to_tensor(np.random.random((10, 5, 1)))  # batch, seq_len, dim
    model = GenericTemporalConvNet(n_filters=(20, 50, 1), problem_type='classification')
    y_hat1 = model(x)
    y_hat2 = model(x)
    np.testing.assert_array_equal(y_hat1, y_hat2)


def test_generic_call_probabilistic():
    """Running inference in training mode should result in random samples being drawn from the distribution of
    model weights.

    This should result in different predictions being made on the same inputs if we run the model multiple times.
    However, if we control the random seed used by Tensorflow, we should be able to reproduce the same behaviour.
    """
    # Create a random number generator to ensures we generate the same inputs each time
    rng = np.random.default_rng(seed=12345)
    x = tf.convert_to_tensor(rng.random(size=(10, 5, 2)))  # batch, seq_len, dim
    # Construct the network
    model = GenericTemporalConvNet(n_filters=(20, 50, 1), problem_type='classification')

    # Reset the random seed and make two predictions (should result in different outputs)
    tf.random.set_seed(12345)
    y_hat1 = model(x, training=True)
    y_hat2 = model(x, training=True)
    # Assert 1 & 2 are not equal
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y_hat1, y_hat2)

    # Reset the random seed again and make another prediction on the same inputs
    tf.random.set_seed(12345)
    y_hat3 = model(x, training=True)
    # Assert 1 & 3 are equal
    np.testing.assert_array_equal(y_hat1, y_hat3)


# def test_full_example():
#     SEQUENCE_LENGTH = 20
#     N_FEATURES = 1
#     N_EXAMPLES = 10000
#     transition_indices = np.random.uniform(low=5., high=16., size=N_EXAMPLES).astype(np.int)
#
#     def generate_labels(t_idx, sequence_len):
#         zeros = np.zeros(t_idx)
#         ones = np.ones(sequence_len - t_idx)
#         return np.expand_dims(np.concatenate((zeros, ones), axis=0).astype(int), axis=1)
#
#     labels = np.array([generate_labels(t_idx, SEQUENCE_LENGTH) for t_idx in transition_indices])
#
#     jitter = 0.2 * np.random.randn(N_EXAMPLES, SEQUENCE_LENGTH, N_FEATURES)
#
#     features = labels + jitter
#
#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     BATCH_SIZE = 256
#     BUFFER_SIZE = 10000
#     dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
#     tcn = GenericTemporalConvNet(n_filters=(10, 20, 50, 10, 1), problem_type='classification')
#     tcn.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
#     tcn.fit(dataset, epochs=10, steps_per_epoch=40)
#     y_hat = tcn.predict(features)
#     print("***************")
#     print(y_hat.min())