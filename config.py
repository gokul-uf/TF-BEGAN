import tensorflow as tf

class Config:
	batch_size = 16
	n = 13 # TODO: change

	embedding_dim = 64
	z_low_limit = -1
	z_high_limit = 1

	img_width = 64
	img_height = 64
	num_channel = 3

	lambda_k = 0.001
	gamma = 0.5
	k_0 = 0
	lr = 5e-5

	sample_location = "../samples"
	num_samples = 25
	num_rows = 5 # for the tiling in sampling
	num_cols = 5

	num_epoch = 1000
	sample_epoch = 5 # generate samples every 5 epochs
	dataset_size = 202599 #CelebA dataset size
	data_location = "../data/img_align_celeba/"
