import tensorflow as tf
import numpy as np



def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def custom_conv2d(input_layer, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None,
                 padding='SAME', scope="conv2d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_layer, w,
                            strides=[1, d_h, d_w, 1], padding=padding)
        b = tf.get_variable("b", shape=output_dim, initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(conv, b)
        return conv


def custom_conv1d(input_layer, output_dim, k_w=5, d_w=2, stddev=0.02, in_dim=None,
                  padding='SAME', scope="conv1d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_w, in_dim or input_layer.shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_layer, w,
                            stride=d_w, padding=padding)
        b = tf.get_variable("b", shape=output_dim, initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(conv, b)
        return conv


def custom_fc(input_layer, output_size, scope='Linear',
              in_dim=None, stddev=0.02, bias_start=0.0):
    shape = input_layer.shape
    if len(shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, int(np.prod(shape[1:]))])
    shape = input_layer.shape
    with tf.variable_scope(scope):
        matrix = tf.get_variable("weight",
                                 [in_dim or shape[1], output_size],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.nn.bias_add(tf.matmul(input_layer, matrix), bias)


def custom_NN1d(x, n):
    """ Nearest neighbour (n times) interpolation for 1D m-channel input, x. """
    # We have to use tf.tile because tf doesn't have a np.repeat equivalent(!)
    s = x.get_shape()
    x = tf.tile(x, [1,1,n])
    x = tf.reshape(x, [s[0],s[1]*n,s[2]])
    return x
