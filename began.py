import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_init_conv2d
from layers import upsample_2d, fully_connected, conv2d, l1_norm
from config import Config as conf

def decoder(input, name, reuse = False):
	with tf.variable_scope(name, reuse = reuse) as scope:
		fc_op = fully_connected(input, num_output = 8*8*conf.n, name = "fc")
		reshape_op = tf.reshape(fc_op, [None, 8, 8, conf.n])
		
		conv_1 = conv2d(input = reshape_op, filter_shape = [3, 3, conf.n, conf.n], name = "conv_1")
		conv_2 = conv2d(input = conv_1,     filter_shape = [3, 3, conf.n, conf.n], name = "conv_2")
		ups_1  = upsample_2d(conv_2, size = [16, 16], name = "ups_1")

		conv_3 = conv2d(input =  ups_1, filter_shape = [3, 3, conf.n, conf.n], name = "conv_3")
		conv_4 = conv2d(input = conv_3, filter_shape = [3, 3, conf.n, conf.n], name = "conv_4")
		ups_2  = upsample_2d(conv_4, size = [32, 32], name = "ups_2")		

		conv_5 = conv2d(input =  ups_2, filter_shape = [3, 3, conf.n, conf.n], name = "conv_5")
		conv_6 = conv2d(input = conv_5, filter_shape = [3, 3, conf.n, conf.n], name = "conv_6")
		ups_3  = upsample_2d(conv_6, size = [64, 64], name = "ups_3")

		conv_7 = conv2d(input =  ups_3, filter_shape = [3, 3, conf.n, conf.n], name = "conv_7")
		conv_8 = conv2d(input = conv_7, filter_shape = [3, 3, conf.n, conf.n], name = "conv_8")
		conv_9 = conv2d(input = conv_8, filter_shape = [3, 3, conf.n, 3], name = "conv_9")

		return conv_9


def encoder(input, name, reuse = False):
	with tf.variable_scope(name, reuse = reuse) as scope:
		conv_0 = conv2d(input = input,  filter_shape = [3, 3, 3, conf.n],      name = "conv_0")
		conv_1 = conv2d(input = conv_0, filter_shape = [3, 3, conf.n, conf.n],      name = "conv_1")
		conv_2 = conv2d(input = conv_1, filter_shape = [3, 3, conf.n, conf.n], name = "conv_2")
		subs_1 = conv2d(input = conv_2, filter_shape = [3, 3, conf.n, 2*conf.n], strides = (1, 2, 2, 1), name = "subs_1")

		conv_3 = conv2d(input = subs_1, filter_shape = [3, 3, 2*conf.n, 2*conf.n], name = "conv_3")
		conv_4 = conv2d(input = conv_3, filter_shape = [3, 3, 2*conf.n, 2*conf.n], name = "conv_4")
		subs_2 = conv2d(input = conv_4, filter_shape = [3, 3, 2*conf.n, 3*conf.n], strides = (1, 2, 2, 1), name = "subs_2")

		conv_5 = conv2d(input = subs_2, filter_shape = [3, 3, 3*conf.n, 3*conf.n], name = "conv_5")
		conv_6 = conv2d(input = conv_5, filter_shape = [3, 3, 3*conf.n, 3*conf.n], name = "conv_6")
		subs_3 = conv2d(input = conv_6, filter_shape = [3, 3, 3*conf.n, 4*conf.n], strides = (1, 2, 2, 1), name = "subs_3")

		conv_7 = conv2d(input = subs_3, filter_shape = [3, 3, 4*conf.n, 4*conf.n], name = "conv_7")
		conv_8 = conv2d(input = conv_7, filter_shape = [3, 3, 4*conf.n, 4*conf.n], name = "conv_8")
		reshape_op = tf.reshape(conv_8 ,[None, 8*8*4*conf.n])
		fc_op = fully_connected(reshape_op, num_output = conf.embedding_dim, name = "fc")

		return fc_op