import tensorflow as tf
import numpy as np


def w(shape, stddev=0.01):
    for i in range(len(shape)):
        shape[i] = int(shape[i])
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def b(shape, const=0.1):
    return tf.Variable(tf.constant(const, shape=shape))


def conv2d(x, W, b, stride=1, padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')
