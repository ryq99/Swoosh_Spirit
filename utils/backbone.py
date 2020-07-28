import tensorflow as tf
from utils.block import conv_block

def cspdarknet53_tiny():

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
