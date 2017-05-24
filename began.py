import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_init_conv2d
# import tensorflow.image.resize_nearest_neighbor as resize_nn
# from tensorflow.nn import conv2d, elu
from config import Config as conf



input_embedding = tf.placeholder(tf.float32, [conf.batch_size, conf.embedding_dim], 
									name = "input_embedding")

# Generator
# Declaring weights
w_fc_gen = tf.get_variable("w_fc_gen", shape = [conf.embedding_dim, 8*8*conf.n],
							initializer = xavier_init())
w_conv_1_gen = tf.get_variable("w_conv_1_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_2_gen = tf.get_variable("w_conv_2_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_3_gen = tf.get_variable("w_conv_3_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_4_gen = tf.get_variable("w_conv_4_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_5_gen = tf.get_variable("w_conv_5_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_6_gen = tf.get_variable("w_conv_6_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_7_gen = tf.get_variable("w_conv_7_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_8_gen = tf.get_variable("w_conv_8_gen", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_9_gen = tf.get_variable("w_conv_9_gen", shape = [3, 3, conf.n, 3],
							initializer = xavier_init_conv2d())


# operations for generator / decoder
# TODO asserts
op_fc_gen = tf.reshape(tf.matmul(input_embedding, w_fc_gen), [conf.batch_size, 8, 8, conf.n], name = "fc_gen")
op_conv_1_gen = tf.nn.elu(tf.nn.conv2d(input = op_fc_gen, filter = w_conv_1_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_1_gen")
op_conv_2_gen = tf.nn.elu(tf.nn.conv2d(input = op_conv_1_gen, filter = w_conv_2_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_2_gen")

op_upsample_1_gen  = tf.image.resize_nearest_neighbor(op_conv_2_gen, size = [16, 16], name = "upsample_1_gen") 

op_conv_3_gen = tf.nn.elu(tf.nn.conv2d(input = op_upsample_1_gen, filter = w_conv_3_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_3_gen")
op_conv_4_gen = tf.nn.elu(tf.nn.conv2d(input = op_conv_3_gen, filter = w_conv_4_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_4_gen")

op_upsample_2_gen  = tf.image.resize_nearest_neighbor(op_conv_4_gen, size = [32, 32], name = "upsample_2_gen")

op_conv_5_gen = tf.nn.elu(tf.nn.conv2d(input = op_upsample_2_gen, filter = w_conv_5_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_5_gen")
op_conv_6_gen = tf.nn.elu(tf.nn.conv2d(input = op_conv_5_gen, filter = w_conv_6_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_6_gen")

op_upsample_3_gen  = tf.image.resize_nearest_neighbor(op_conv_6_gen, size = [64, 64], name = "upsample_3_gen")

op_conv_7_gen = tf.nn.elu(tf.nn.conv2d(input = op_upsample_3_gen, filter = w_conv_7_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_7_gen")
op_conv_8_gen = tf.nn.elu(tf.nn.conv2d(input = op_conv_7_gen, filter = w_conv_8_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_8_gen")
op_conv_9_gen = tf.nn.elu(tf.nn.conv2d(input = op_conv_7_gen, filter = w_conv_9_gen, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_9_gen")
assert op_conv_9_gen.shape == (conf.batch_size, 64, 64, 3)

# Encoder 
# Declaring weights

# Decoder
# Declaring weights
w_fc_dec = tf.get_variable("w_fc_dec", shape = [conf.embedding_dim, 8*8*conf.n],
							initializer = xavier_init())
w_conv_1_dec = tf.get_variable("w_conv_1_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_2_dec = tf.get_variable("w_conv_2_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_3_dec = tf.get_variable("w_conv_3_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_4_dec = tf.get_variable("w_conv_4_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_5_dec = tf.get_variable("w_conv_5_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_6_dec = tf.get_variable("w_conv_6_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())

w_conv_7_dec = tf.get_variable("w_conv_7_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_8_dec = tf.get_variable("w_conv_8_dec", shape = [3, 3, conf.n, conf.n],
							initializer = xavier_init_conv2d())
w_conv_9_dec = tf.get_variable("w_conv_9_dec", shape = [3, 3, conf.n, 3],
							initializer = xavier_init_conv2d())

# operations for decoder
# TODO asserts
op_fc_dec = tf.reshape(tf.matmul(input_embedding, w_fc_dec), [conf.batch_size, 8, 8, conf.n], name = "fc_dec") # TODO: Change the input
op_conv_1_dec = tf.nn.elu(tf.nn.conv2d(input = op_fc_dec, filter = w_conv_1_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_1_dec")
op_conv_2_dec = tf.nn.elu(tf.nn.conv2d(input = op_conv_1_dec, filter = w_conv_2_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_2_dec")

op_upsample_1_dec  = tf.image.resize_nearest_neighbor(op_conv_2_dec, size = [16, 16], name = "upsample_1_dec") 

op_conv_3_dec = tf.nn.elu(tf.nn.conv2d(input = op_upsample_1_dec, filter = w_conv_3_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_3_dec")
op_conv_4_dec = tf.nn.elu(tf.nn.conv2d(input = op_conv_3_dec, filter = w_conv_4_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_4_dec")

op_upsample_2_dec  = tf.image.resize_nearest_neighbor(op_conv_4_dec, size = [32, 32], name = "upsample_2_dec")

op_conv_5_dec = tf.nn.elu(tf.nn.conv2d(input = op_upsample_2_dec, filter = w_conv_5_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_5_dec")
op_conv_6_dec = tf.nn.elu(tf.nn.conv2d(input = op_conv_5_dec, filter = w_conv_6_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_6_dec")

op_upsample_3_dec  = tf.image.resize_nearest_neighbor(op_conv_6_dec, size = [64, 64], name = "upsample_3_dec")

op_conv_7_dec = tf.nn.elu(tf.nn.conv2d(input = op_upsample_3_dec, filter = w_conv_7_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_7_dec")
op_conv_8_dec = tf.nn.elu(tf.nn.conv2d(input = op_conv_7_dec, filter = w_conv_8_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_8_dec")
op_conv_9_dec = tf.nn.elu(tf.nn.conv2d(input = op_conv_7_dec, filter = w_conv_9_dec, strides = (1, 1, 1, 1), padding = "SAME"), name = "conv_9_dec")
assert op_conv_9_dec.shape == (conf.batch_size, 64, 64, 3)


