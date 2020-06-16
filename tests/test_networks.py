from ptcn.networks import TemporalConvNet
import tensorflow as tf

def test_temporalconvnet():
    tcn = TemporalConvNet(
        n_filters=(5,5,1)
    )
    tcn.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    tcn.build(input_shape=(1,5,1))
    print(tcn.summary())