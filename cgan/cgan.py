# example of training an conditional gan
import numpy as np
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate

import pickle

import time
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

import pandas as pd


SELECTED_CLASSES = [7, 12, 14, 15]

TIMES = []

def denormalize(power_predictions, class_predictions, csv_path='../data/data_collect_maxmin.csv', filename='gan_results.pickle'):
	df = pd.read_csv(csv_path)
	final_arr = []
	# results is a list of lists -- each entry contains building_type id, followed
	# by the normalized val.
	for idx, power_prediction in enumerate(power_predictions):
		building_type = class_predictions[idx]
		row = df.values[SELECTED_CLASSES[building_type]]
		max_val = row[1]
		min_val = row[2]
		denormalized_val = power_prediction * (max_val - min_val) + (max_val + min_val)
		denormalized_val /= 2
		final_arr.append([SELECTED_CLASSES[building_type], denormalized_val])

	# convert to pickle file for now.
	print(final_arr[0][1].shape)
	with open(filename, 'wb') as handle:
		pickle.dump(final_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return final_arr


def prediction(zs, classes, g_model, filename):
	# predict normalized value of generator with latent space as input.
	results = []
	normalized_val = g_model.predict([zs, classes])
	# predict class with normalized val
	class_predictions = classes
	power_predictions = normalized_val.reshape(normalized_val.shape[0], normalized_val.shape[1])
	preds = {}
	for prediction in class_predictions:
		if prediction in preds:
			preds[prediction] += 1
		else:
			preds[prediction] = 1
	print(np.unique(class_predictions))
	print(preds)
	final = denormalize(power_predictions, class_predictions, filename=filename)


# define the standalone discriminator model
def define_discriminator(in_shape=(744, 1, 1), n_classes=4):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input, embedding dim of 50
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,1), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Conv2D(128, (3,3), strides=(2, 1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps

	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=4):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 186
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((186, 1, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 186
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((186, 1, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,1), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,1), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# out_layer = Reshape((744, 1, 1))(out_layer)

	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def load_real_samples_grid(num_types=4, num_per_type=100):
	# load the REAL data.
	f = open('../data/data_collect_select_equal.csv','r')
	lines = f.readlines()
	f.close()
	X_temp = []
	y_temp = []
	for i in range(1,len(lines)):
		y_temp.append(int(lines[i].split(',')[-1].replace('\n','')))
		X_temp_temp = []
		for j in range(1,len(lines[i].split(','))-1):
			val = float(lines[i].split(',')[j].replace('\n',''))
			X_temp_temp.append(val)
		X_temp.append(X_temp_temp)
	trainy = asarray(y_temp)
	trainX = asarray(X_temp)
	# expand to 3d, e.g. add channels
	X1 = expand_dims(trainX, axis=-1)
	X = expand_dims(X1, axis=-1)
	print(X.shape, trainy.shape)

	dataset = [X, trainy]
	new_X = []
	new_trainy = []

	types_dict = {}
	for i in range(num_types):
		types_dict[i] = 0

	current_type_idx = 0
	for idx, val in enumerate(dataset[0]):
		if dataset[1][idx] == current_type_idx:
			types_dict[current_type_idx] += 1
			new_X.append(val)
			new_trainy.append(dataset[1][idx])
			if types_dict[current_type_idx] == num_per_type:
				current_type_idx += 1
	print(np.array(new_X).shape, 'shape')

	return [np.array(new_X), np.array(new_trainy)]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=310, n_batch=32,
          num_classes=4, amt=100):
	total_time = 0
	bat_per_epo = int(np.array(dataset[0]).shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		begin = time.time()

		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes=num_classes)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch, n_classes=num_classes)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			# summarize performance after every epoch
		print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

		total_time += (time.time() - begin)
		if (i+1) % 50 == 0:
			g_model.save(f'./h5/cgan_trainsize{amt}_epochs{i+1}.h5')


# freq_dict={7: 400, 12: 4500, 14: 800, 15:400,
						  			#  4: 40, 5: 250, 9: 200, 10: 100}
def summarize_performance(step, g_model, d_model, latent_dim, dataset, n_samples=6100,
                          num_classes=4,
						  train_size=100,
						  freq_dict={7: 400, 12: 4500, 14: 800, 15: 400}):

	dataset_len = len(dataset[0])
	X_real, y_real = dataset
	[X_fake, y_fake], _ = generate_fake_samples(g_model, latent_dim, dataset_len, n_classes=num_classes)
	# y1 = ones((2800, 1))
	# y0 = zeros((2800, 1))

	y1 = ones((dataset_len, 1))
	y0 = zeros((dataset_len, 1))
	num_gen = 0
	for val in freq_dict:
		num_gen += (freq_dict[val])
	print(num_gen, ' will be generated')
	zs, _ = generate_latent_points(latent_dim, num_gen, n_classes=num_classes)
    # run some prediction
	# compute classes
	classes = []

	idx = 0
	for val in freq_dict:
		print(val, freq_dict[val], idx)
		for j in range(freq_dict[val]):
			classes.append(idx)
		idx += 1

	# run prediction
	# prediction(zs, np.array(classes), g_model, filename=f'cgan_results_trainsize{train_size}_epoch{step}.pickle')

	# d_loss_real = d_model.evaluate([X_real, y_real], y1, verbose=0)[0]
	# d_loss_real_wrong = d_model.evaluate([X_real, y_real], y0, verbose=0)[0]
	# d_loss_fake = d_model.evaluate([X_fake, y_fake], y0, verbose=0)[0]
	# d_loss_fake_wrong = d_model.evaluate([X_fake, y_fake], y1, verbose=0)[0]
	# print('Discriminator Loss: Real: %.3f; Real_wrong: %.3f; Fake: %.3f; Fake_wrong: %.3f' % (d_loss_real, d_loss_real_wrong, d_loss_fake, d_loss_fake_wrong))
	# save the generator model


def main(types, num_train):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)

    # size of the latent space
	latent_dim = 1000
	# create the discriminator
	num_classes = len(types)
	d_model = define_discriminator(n_classes=num_classes)
	# create the generator
	g_model = define_generator(latent_dim, n_classes=num_classes)
	# create the gan
	gan_model = define_gan(g_model, d_model)
	# load image data
	dataset = load_real_samples_grid(num_types=num_classes, num_per_type=num_train)
	# train model
	train(g_model, d_model, gan_model, dataset, latent_dim,
			amt=num_train, num_classes=num_classes,
			n_epochs=2000)
