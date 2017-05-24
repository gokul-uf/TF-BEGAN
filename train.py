import tensorflow as tf
import numpy as np

def generate_noise_batch(size):
	pass


if __name__ == '__main__':
	input_embedding = tf.placeholder(tf.float32, [conf.batch_size, conf.embedding_dim], name = "input_embedding")
	input_image     = tf.placeholder(tf.float32, [conf.batch_size, conf.img_width, conf.img_height, conf.num_channel], name = "input_image")
	k_t 			= tf.placeholder(tf.float32, shape=(), name = "k_t")

	enc_orig_image = encoder(input_image, "encoder")
	dec_orig_image = decoder(enc_orig_image, "decoder")

	gen_image = decoder(input_embedding, "generator")
	enc_gen_image = encoder(gen_image, "encoder", reuse = True)
	dec_gen_image = decoder(enc_gen_image, "decoder", reuse = True)

	l_x   = l1_norm(tf.subtract(input_image, dec_orig_image),  "l_x")
	l_g_d = l1_norm(tf.subtract(gen_image,   dec_gen_image),   "l_g_d")
	# l_g_g = #TODO
	# m_global = l_x + tf.abs(tf.subtract(tf.mul(gamma, l_x), l_g_g))
	# what's Z_g?
	# what's lambda and gamma?

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# get the data