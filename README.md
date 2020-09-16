# ptcn

Tensorflow (2.x) implementation of a Temporal Convolutional Network architecture, with a probabilistic twist.

This project indulges a couple of curiosities:

1. Working with convolutional sequence-to-sequence models a la [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
2. Adding a bayesian twist to the network a la [Bayesian Segnet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/abs/1511.02680)

This implementation has been inspired by other projects, including:
- https://github.com/locuslab/TCN
- https://github.com/Baichenjia/Tensorflow-TCN
- https://github.com/philipperemy/keras-tcn

## Usage

### Detecting events as shifts between regimes

Sometimes I find myself working on problems where a variable is expected to shift from one regime to another at a point in time, and we want to detect when that shift happened. If we're working with noisy data, detecting that event with confidence can be tricky.

Sequence-to-sequence learning, and TCNs in particular allow us to produce labels for each time step whilst taking multiple time-steps into account when labelling each individual time step. What's really powerful about using a convolutional architecture is that we can train one model, but then apply it to time series of different lengths at inference time.

In this example, we'll create some synthetic data to describe this type of problem, and then look at the TCN's ability to detect the occurrence of the regime-switch.


```python
import ptcn
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
```

### Create a synthetic dataset

The objective in this exercise is to demonstrate the ability for the ptcn to learn coherent classification labels for each element of a sequence.

I am particularly interested in estimating labels for sequences that demonstrate a path-dependent state transition; ie. the labels of later elements depend on what has gone before, not just the feature values at that timestamp.

I create this path dependence by creating sequences whose labels pass from an initial phase in which all elements belong to category 1 (labelled as 0), and a second phase in which all elements belong to category 2 (labelled as 1). Each sample varies according to where in the sequence the transition from phase 1 to phase 2 occurs.

**Config**

To begin with, we'll specify some of the properties of our dataset.

`SEQUENCE_LENGTH` defines the number of time steps to include in each sequence

`N_FEATURES` defines the number of feature variables to create

`N_EXAMPLES` defines the size of the synthetic dataset - the number of examples to create


```python
SEQUENCE_LENGTH = 20
N_FEATURES = 1
N_EXAMPLES = 10000
```

### Randomly sample transition points

We begin by randomly sampling the index position in each example at which the transition from phase 1 to phase 2 will occur. We'll use this to generate appropriate feature & target values.

The `low` & `high` args set a boundary on where in the sequence these transitions are permitted to occur (we choose the constrain the transitions to occur between the 5th & the 15th index position).


```python
transition_indices = np.random.uniform(low=5., high=16., size=N_EXAMPLES).astype(np.int)
```

### Generate classification labels

We use these 'transition indices' to generate the classification labels for our dataset.

The results in an array with dimensions: (N_EXAMPLES, SEQUENCE_LENGTH, N_FEATURES)


```python
def generate_labels(t_idx, sequence_len):
    zeros = np.zeros(t_idx)
    ones = np.ones(sequence_len - t_idx)
    return np.expand_dims(np.concatenate((zeros, ones), axis=0), axis=1)
```


```python
labels = np.array([generate_labels(t_idx, SEQUENCE_LENGTH) for t_idx in transition_indices])
print(labels.shape)
```

    (10000, 20, 1)


### Generate feature values

In order to create features that can be used to predict these labels, we duplicate the labels and add random jitter to them.


```python
jitter = 0.1 * np.random.randn(N_EXAMPLES, SEQUENCE_LENGTH, N_FEATURES)
jitter.shape
```




    (10000, 20, 1)




```python
def generate_step_features(t_idx, sequence_len):
    zeros = np.zeros(t_idx)
    ones = np.ones(sequence_len - t_idx)
    return np.expand_dims(np.concatenate((zeros, ones), axis=0), axis=1)

step_features = np.array([generate_step_features(t_idx, SEQUENCE_LENGTH) for t_idx in transition_indices])
step_features.shape
```




    (10000, 20, 1)




```python
def generate_relu_features(t_idx, sequence_len):
    zeros = np.zeros(t_idx)
    # We have to round because the floats are otherwise not exactly divisible by 0.1 and we get inconsistent sequence lengths
    ones = np.ones(sequence_len - t_idx) * np.arange(0.1, round(0.1 * (1 + sequence_len - t_idx), 2), 0.1)
    return np.expand_dims(np.concatenate((zeros, ones), axis=0), axis=1)

relu_features = np.array([generate_relu_features(t_idx, SEQUENCE_LENGTH) for t_idx in transition_indices])
relu_features.shape
```




    (10000, 20, 1)




```python
features = relu_features + jitter
features.shape
```




    (10000, 20, 1)



### Visualise the data


```python
fig = go.Figure()

for i in range(10):
    fig.add_trace(
        go.Scatter(
            x=list(range(len(features[i,:,:]))), y=np.squeeze(features[i,:,:]),
            mode='lines+markers',
            name=f'Series {i}'
        )
    )
    
fig.show()
```


### Split the data into training & test


```python
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
BATCH_SIZE = 256
BUFFER_SIZE = 10000
dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
```

## Initialise the model


```python
from ptcn.networks import GenericTemporalConvNet
```


```python
tcn = GenericTemporalConvNet(n_filters=(10, 20, 50, 10, 1), problem_type='classification')
```


```python
loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0.,
    name='binary_crossentropy'
)
```


```python
tcn.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=loss)
```


```python
tcn.fit(dataset, epochs=60, steps_per_epoch=40)
```

    Epoch 1/60
    WARNING:tensorflow:Layer generic_temporal_conv_net is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    40/40 [==============================] - 1s 35ms/step - loss: 0.7756
    Epoch 2/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.7546
    Epoch 3/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.7344
    Epoch 4/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.7268
    Epoch 5/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.7159
    Epoch 6/60
    40/40 [==============================] - 1s 37ms/step - loss: 0.7108
    Epoch 7/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.6526
    Epoch 8/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.6004
    Epoch 9/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.5503
    Epoch 10/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.5078
    Epoch 11/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.4623
    Epoch 12/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.4088
    Epoch 13/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.3730
    Epoch 14/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.3532
    Epoch 15/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.3426
    Epoch 16/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.3235
    Epoch 17/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.3044
    Epoch 18/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2963
    Epoch 19/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2763
    Epoch 20/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.2652
    Epoch 21/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2636
    Epoch 22/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2714
    Epoch 23/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2617
    Epoch 24/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2585
    Epoch 25/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2573
    Epoch 26/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2480
    Epoch 27/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2441
    Epoch 28/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.2393
    Epoch 29/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2356
    Epoch 30/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2324
    Epoch 31/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2253
    Epoch 32/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2207
    Epoch 33/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2128
    Epoch 34/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2108
    Epoch 35/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.2074
    Epoch 36/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2036
    Epoch 37/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2019
    Epoch 38/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1971
    Epoch 39/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1948
    Epoch 40/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1962
    Epoch 41/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1923
    Epoch 42/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1917
    Epoch 43/60
    40/40 [==============================] - 1s 37ms/step - loss: 0.1906
    Epoch 44/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1884
    Epoch 45/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1859
    Epoch 46/60
    40/40 [==============================] - 1s 37ms/step - loss: 0.1847
    Epoch 47/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1826
    Epoch 48/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1807
    Epoch 49/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1799
    Epoch 50/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1778
    Epoch 51/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1752
    Epoch 52/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1748
    Epoch 53/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1742
    Epoch 54/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1724
    Epoch 55/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1711
    Epoch 56/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1706
    Epoch 57/60
    40/40 [==============================] - 1s 37ms/step - loss: 0.1708
    Epoch 58/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1707
    Epoch 59/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1691
    Epoch 60/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1693





    <tensorflow.python.keras.callbacks.History at 0x7fa248542af0>



## Evaluation


```python
predictions = tcn(features)
```


```python
p_predictions_1 = tcn(features, training=True)
```


```python
p_predictions_2 = tcn(features, training=True)
```


```python
for i in range(15):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels[i,:,:]))), y=np.squeeze(labels[i,:,:]),
            mode='lines+markers',
            name=f'Labels {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(features[i,:,:]))), y=np.squeeze(features[i,:,:]),
            mode='lines+markers',
            name=f'Features {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(predictions[i,:,:]))), y=np.squeeze(predictions[i,:,:]),
            mode='lines+markers',
            name=f'Predictions {i}'
        )
    )
    fig.show()
```


## Training with non-causal conv


```python
tcn2 = GenericTemporalConvNet(n_filters=(10, 20, 50, 10, 1), padding_type="same", problem_type='classification')
```


```python
tcn2.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=loss)
```


```python
tcn2.fit(dataset, epochs=60, steps_per_epoch=40)
```

    Epoch 1/60
    WARNING:tensorflow:Layer generic_temporal_conv_net_1 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    40/40 [==============================] - 2s 38ms/step - loss: 0.6573
    Epoch 2/60
    40/40 [==============================] - 2s 38ms/step - loss: 0.4354
    Epoch 3/60
    40/40 [==============================] - 1s 37ms/step - loss: 0.3758
    Epoch 4/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.3619
    Epoch 5/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.3178
    Epoch 6/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2840
    Epoch 7/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2690
    Epoch 8/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2449
    Epoch 9/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2369
    Epoch 10/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.2440
    Epoch 11/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2293: 0s - loss: 
    Epoch 12/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2240
    Epoch 13/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.2099
    Epoch 14/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.2006
    Epoch 15/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.2054
    Epoch 16/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.2059
    Epoch 17/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1966
    Epoch 18/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1800
    Epoch 19/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1660
    Epoch 20/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1673
    Epoch 21/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1632
    Epoch 22/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1609
    Epoch 23/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1636
    Epoch 24/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1589
    Epoch 25/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1552
    Epoch 26/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1728
    Epoch 27/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1496
    Epoch 28/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.1435
    Epoch 29/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1417
    Epoch 30/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1305
    Epoch 31/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.1408
    Epoch 32/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.1375
    Epoch 33/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1401
    Epoch 34/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1323
    Epoch 35/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1333
    Epoch 36/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1307
    Epoch 37/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1264
    Epoch 38/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1225
    Epoch 39/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1224
    Epoch 40/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.1212
    Epoch 41/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1238
    Epoch 42/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1242
    Epoch 43/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1210
    Epoch 44/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1158
    Epoch 45/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1168
    Epoch 46/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1201
    Epoch 47/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1198
    Epoch 48/60
    40/40 [==============================] - 1s 32ms/step - loss: 0.1215
    Epoch 49/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1185
    Epoch 50/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1105
    Epoch 51/60
    40/40 [==============================] - 1s 34ms/step - loss: 0.1103
    Epoch 52/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1092
    Epoch 53/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1041
    Epoch 54/60
    40/40 [==============================] - 1s 36ms/step - loss: 0.1100
    Epoch 55/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.1069
    Epoch 56/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1014
    Epoch 57/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1018
    Epoch 58/60
    40/40 [==============================] - 2s 38ms/step - loss: 0.1004
    Epoch 59/60
    40/40 [==============================] - 1s 35ms/step - loss: 0.0992
    Epoch 60/60
    40/40 [==============================] - 1s 33ms/step - loss: 0.1011





    <tensorflow.python.keras.callbacks.History at 0x7fa2300a82b0>




```python
predictions2 = tcn2(features)
```


```python
for i in range(15):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels[i,:,:]))), y=np.squeeze(labels[i,:,:]),
            mode='lines+markers',
            name=f'Labels {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(features[i,:,:]))), y=np.squeeze(features[i,:,:]),
            mode='lines+markers',
            name=f'Features {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(predictions2[i,:,:]))), y=np.squeeze(predictions2[i,:,:]),
            mode='lines+markers',
            name=f'Predictions {i}'
        )
    )
    fig.show()
```


## Probabilistic interpretation

For each input, duplicate x 15 & run inference in training mode so we sample from different weights each time

Take mean of estimates for each timestamp for point estimate
Take st. dev. of estimates for each timestamp for confidence in each point estimate

Plot mean & confidence interval


```python
def plot_prob_pred(labels, features, mean_est, est_std):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels))), y=np.squeeze(labels),
            mode='lines+markers',
            name=f'Labels {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(features))), y=np.squeeze(features),
            mode='lines+markers',
            name=f'Features {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_est))), y=np.squeeze(mean_est),
            mode='lines+markers',
            name=f'Mean prediction {i}'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_est))) + list(range(len(mean_est)))[::-1], y=np.clip(np.concatenate((np.squeeze(mean_est + 2*est_std), np.squeeze(mean_est - 2*est_std)[::-1])), a_min=0., a_max=1.),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            mode='lines',
            name=f'95% confidence {i}'
        )
    )       

#     plt.fill_between(list(range(len(mean_est))), np.squeeze(mean_est - 2*est_std), np.squeeze(mean_est + 2*est_std), color='b', alpha=.1)
    
    fig.show()
```


```python
for i in range(15):
    p_prediction = tcn2(np.repeat(features[i:i+1], 15, axis=0), training=True)
    mean_est = np.mean(p_prediction, axis=0)
    est_std = np.std(p_prediction, axis=0)
    plot_prob_pred(labels[i,:,:], features[i,:,:], mean_est, est_std)
```



