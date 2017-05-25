import tensorflow as tf
from began import encoder, decoder
from layers import l1_norm
import numpy as np
from config import Config as conf
from tqdm import tqdm
from PIL import Image
import os

def get_noise_batch(size = [conf.batch_size, conf.embedding_dim]):
	return np.random.uniform(low = conf.z_low_limit, high = conf.z_high_limit, size = size)

def get_img_batch(img_ids, loc = conf.data_location, img_shape = (conf.img_height, conf.img_width)):
	assert len(img_ids) == conf.batch_size
	img_batch = []
	for img_id in img_ids:
		img_file = "{}.jpg".format(str(img_id).zfill(6)) # CelebA images are numbered 000001, 000002...
		img = Image.open(conf.data_location + "/" + img_file)
		img = img.resize((conf.img_width, conf.img_height)) # resize to 64x64
		img_data = np.asarray(img, np.float32)
		img_batch.append(img_data)
	img_batch = np.asarray(img_batch, dtype = np.float32)
	assert img_batch.shape == (conf.batch_size, conf.img_height, conf.img_width, conf.num_channel)
	return (img_batch - 127.) / 127.


if __name__ == '__main__':

	z_d 			= tf.placeholder(tf.float32, [None, conf.embedding_dim], name = "z_d")
	z_g 			= tf.placeholder(tf.float32, [None, conf.embedding_dim], name = "z_g")
	input_image     = tf.placeholder(tf.float32, [None, conf.img_height, conf.img_width,
									 conf.num_channel], name = "input_image")
	k_t 			= tf.placeholder(tf.float32, shape=(), name = "k_t")

	enc_orig_image = encoder(input_image, "encoder")
	dec_orig_image = decoder(enc_orig_image, "decoder")
	l_x   = l1_norm(tf.subtract(input_image, dec_orig_image), "l_x")

	gen_image_z_d   = decoder(z_d, "generator")
	enc_gen_img_z_d = encoder(gen_image_z_d,   "encoder", reuse = True)
	dec_gen_img_z_d = decoder(enc_gen_img_z_d, "decoder", reuse = True)
	l_g_d = l1_norm(tf.subtract(gen_image_z_d,   dec_gen_img_z_d), "l_g_d")

	gen_image_z_g = decoder(z_g, "generator", reuse = True)
	enc_gen_img_z_g = encoder(gen_image_z_g,   "encoder", reuse = True)
	dec_gen_img_z_g = decoder(enc_gen_img_z_g, "decoder", reuse = True)
	l_g_g = l1_norm(tf.subtract(gen_image_z_g,   dec_gen_img_z_g), "l_g_g")

	l_d = tf.reduce_mean(tf.subtract(l_x, tf.multiply(k_t,l_g_d)))
	l_g = tf.reduce_mean(l_g_g)

	d_train = tf.train.AdamOptimizer(learning_rate = conf.lr).minimize(l_d)
	g_train = tf.train.AdamOptimizer(learning_rate = conf.lr).minimize(l_g)

	if not os.path.exists(conf.sample_location):
		os.makedirs(conf.sample_location)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		k_t_input = conf.k_0
		for i in range(conf.num_epoch):
			d_loss = []
			g_loss = []
			epoch_l_x = [] # To update k_t
			epoch_l_g_g = [] # To update k_t
			all_images = list(range(1, conf.dataset_size + 1))
			np.random.shuffle(all_images)
			for j in tqdm(range(0, conf.batch_size * (conf.dataset_size // conf.batch_size), conf.batch_size)):
				img_batch = get_img_batch(all_images[j : j + conf.batch_size])
				z_g_batch = get_noise_batch()
				z_d_batch = get_noise_batch()
				feed_dict = {z_d : z_d_batch, z_g : z_g_batch,
							 input_image : img_batch, k_t: k_t_input}
				_, _, batch_d_loss, batch_g_loss, batch_l_x, batch_l_g_g = sess.run([d_train, g_train, l_d, l_g, l_x, l_g_g],
																				feed_dict = feed_dict)
				d_loss.append(batch_d_loss)
				g_loss.append(batch_g_loss)
				epoch_l_x.append(batch_l_x)
				epoch_l_g_g.append(batch_l_g_g)
			epoch_l_x = np.mean(epoch_l_x)
			epoch_l_g_g = np.mean(epoch_l_g_g)
			g_loss = np.mean(g_loss)
			d_loss = np.mean(d_loss)
			M_global = epoch_l_x + abs(conf.gamma * epoch_l_x - epoch_l_g_g)
			print("Epoch : {}".format(i+1))
			print("M_global: {}, G Loss: {}, D Loss: {}".format(M_global, g_loss, d_loss))
			k_t_input += conf.lambda_k*(conf.gamma*epoch_l_x - epoch_l_g_g)

			if i % conf.sample_epoch == 0:
				print("Sampling Images")
				sample_noise = get_noise_batch(size = [conf.num_samples, conf.embedding_dim])
				gen_images = sess.run(gen_image_z_d, feed_dict = {z_d: sample_noise})
				assert gen_images.shape == (conf.num_samples, conf.img_height, conf.img_width, conf.num_channel)
				gen_images = (gen_images * 127.) + 127.
				gen_images = [Image.fromarray(gen_images[i].astype(np.uint8)) for i in range(conf.num_samples)] # Image expects uint8 [0, 255]
				sample_images = Image.new("RGB", (conf.num_cols*conf.img_width, conf.num_rows*conf.img_height))
				for row in range(conf.num_rows):
					for col in range(conf.num_cols):
						idx = row*conf.num_cols + col
						sample_images.paste(gen_images[idx], (col*conf.img_width, row*conf.img_height))
				sample_images.save(conf.sample_location + "/" + "epoch_{}.jpg".format(i))
