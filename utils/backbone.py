import tensorflow as tf
from utils.block import conv_block, route_group

def cspdarknet53_tiny(inputs):
    x = conv_block(inputs, channels=32, downsample=True)
    x = conv_block(x, channels=64, downsample=True)
    x = conv_block(x, channels=64)
    route = x
    x = route_group(x, num_groups=2, group_id=1)
    x = conv_block(x, channels=32)
    route_1 = x
    x = conv_block(x, channels=32)
    x = tf.concat([x, route_1], axis=-1)
    x = conv_block(x, channels=64, kernel_size=1)
    x = tf.concat([route, x], axis=-1)
    x = tf.keras.layers.MaxPool2D(pool_size=2, stride=2, padding='same')(x)

    








def darknet53_tiny(inputs):
    x = conv_block(inputs, channels=16, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=32, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=64, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=128, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=256, kernel_size=(3, 3))
    route_1 = x
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=512, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=1024, kernel_size=(3, 3))

    return route_1, x
