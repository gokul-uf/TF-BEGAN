import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_init_conv2d


def upsample_2d(input, size, name):
    with tf.variable_scope(name) as scope:
        return tf.image.resize_nearest_neighbor(input, size=size, name=name)


def fully_connected(input, num_output, name):
    # does input*weight, expects 2D input
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable(
            "weight",
            shape=[input.shape.as_list()[1], num_output],
            initializer=xavier_init())
        return tf.matmul(input, weight)


def conv2d(input,
           filter_shape,
           strides=(1, 1, 1, 1),
           activation=tf.nn.elu,
           pad="SAME",
           name=None):
    with tf.variable_scope(name) as scope:
        filter = tf.get_variable(
            "filter", shape=filter_shape, initializer=xavier_init_conv2d())
        conv_op = tf.nn.conv2d(
            input=input, filter=filter, strides=strides, padding=pad)
        return activation(conv_op)


def l1_norm(input, name):
    # takes a batch of n-dimensional tensor and returns a 2D tensor of l1 norms
    with tf.variable_scope(name) as scope:
        return tf.reduce_sum(
            tf.abs(input), axis=list(range(1, len(input.shape))))
