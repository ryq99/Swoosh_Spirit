import tensorflow as tf


def conv_block(inputs, channels, kernel_size=(3, 3), activation='relu', downsample=False, bn=True):
    if downsample:
        strides = (2, 2)
        padding = 'valid'
        inputs = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(inputs)
    else:
        strides = (1, 1)
        padding = 'same'
    inputs = tf.keras.layers.Conv2D(filters=channels,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=not bn,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.0005),
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.constant_initializer(0.))(inputs)
    if bn:
        inputs = tf.keras.layers.BatchNormalization()(inputs)
    if activation == 'relu':
        outputs = tf.keras.activations.relu(inputs,
                                           alpha=0,
                                           threshold=0)
    if activation == 'leaky':
        outputs = tf.keras.activations.relu(inputs,
                                            alpha=0.1,
                                            threshold=0)
    return outputs


def upsample(inputs):
    outputs = tf.image.resize(inputs,
                              size=(inputs.shape[1] * 2, inputs.shape[2] * 2),
                              method=tf.image.ResizeMethod.BILINEAR)
    return outputs

