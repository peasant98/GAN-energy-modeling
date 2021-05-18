# example of semi-supervised gan for mnist
from keras.utils.np_utils import normalize
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import reshape
import numpy as np
from numpy.random import randn
from numpy.random import uniform
from numpy.random import randint
from keras.datasets.mnist import load_data
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
from keras.layers import Lambda
from keras.layers import Activation
from keras.utils import np_utils
from matplotlib import pyplot
from keras import backend
import tensorflow as tf
tf.executing_eagerly()

import pickle

import time
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

import pandas as pd

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

SELECTED_CLASSES = [7, 12, 14, 15]
TIMES = []


def denormalize(power_predictions, class_predictions, csv_path='data_collect_maxmin.csv', filename='gan_results.pickle'):
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
	with open(filename, 'wb') as handle:
		pickle.dump(final_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return final_arr


def prediction(zs, c_model, g_model, filename):
	# predict normalized value of generator with latent space as input.
	results = []
	normalized_val = g_model.predict(zs)
	# predict class with normalized val
	building_type_id = c_model.predict(normalized_val)
	class_predictions = np.argmax(building_type_id, axis=-1)
	power_predictions = normalized_val.reshape(normalized_val.shape[0], normalized_val.shape[1])
	preds = {}
	for prediction in class_predictions:
		if prediction in preds:
			preds[prediction] += 1
		else:
			preds[prediction] = 1
	print(preds)
	final = denormalize(power_predictions, class_predictions, filename=filename)


# custom activation function
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(744,1,1), n_classes=18):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,1), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output layer nodes
	fe = Dense(n_classes)(fe)
	# supervised output
	c_out_layer = Activation('softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	# unsupervised output
	d_out_layer = Lambda(custom_activation)(fe)
	# define and compile unsupervised discriminator model
	d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return d_model, c_model


# define the standalone generator model
def define_generator(latent_dim):
	# image generator inputpandas
	in_lat = Input(shape=(latent_dim,))
	# foundation for image
	n_nodes = 23808
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((186, 1, 128))(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,1), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,1), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model(in_lat, out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect image output from generator as input to discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and outputting a classification
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def load_real_samples_grid(num_types=4, num_per_type=100):
	# load the REAL data.
	f = open('./data_collect_select_equal.csv','r')
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

	return [np.array(new_X), np.array(new_trainy)]

# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=1000, n_classes=18):
	X, y = dataset
	'''
	X_list, y_list = list(), list()
	n_per_class = int(n_samples / n_classes)
	for i in range(n_classes):
		# get all images for this class
		X_with_class = X[y == i]
		# choose random instances
		ix = randint(0, len(X_with_class), n_per_class)
		# add to list
		[X_list.append(X_with_class[j]) for j in ix]
		[y_list.append(i) for j in ix]
	return asarray(X_list), asarray(y_list)
	'''
	return asarray(X), asarray(y)

# select real samples
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
def generate_latent_points_grid(latent_dim, n_samples):
	# generate points in the latent space
	#z_input = uniform(0,1,latent_dim * n_samples)
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples_grid(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points_grid(latent_dim, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, c_model, latent_dim, dataset, n_samples=40,
						  num_train=400):
	# prepare fake examples
	X, _ = generate_fake_samples_grid(g_model, latent_dim, n_samples)

	X_real, y_real = dataset
	X, _ = generate_fake_samples_grid(g_model, latent_dim, n_samples)

	y1 = ones((n_samples, 1))
	y0 = zeros((n_samples, 1))

	for o in range(6):
		zs = generate_latent_points_grid(latent_dim, int(1600*3.14))
		# run some prediction
		prediction(zs, c_model, g_model, filename=f'sgan_results_trainsize{num_train}_epoch{step}_iter{o}.pickle')

	c_loss, acc = c_model.evaluate(X_real, y_real, verbose=0)
	d_loss_real = d_model.evaluate(X_real, y1, verbose=0)
	d_loss_real_wrong = d_model.evaluate(X_real, y0, verbose=0)
	d_loss_fake = d_model.evaluate(X, y0, verbose=0)
	d_loss_fake_wrong = d_model.evaluate(X, y1, verbose=0)
	print('Classifier Loss: %.3f' % c_loss)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	print('Discriminator Loss: Real: %.3f; Real_wrong: %.3f; Fake: %.3f; Fake_wrong: %.3f' % (d_loss_real, d_loss_real_wrong, d_loss_fake, d_loss_fake_wrong))

	# save the generator model
	filename2 = 'g_model_%04d.h5' % (step+1)
	# g_model.save(filename2)
	# save the classifier model
	filename3 = 'c_model_%04d.h5' % (step+1)
	# c_model.save(filename3)
	accurac_val = 'Classifier Loss: %.3f; ' % c_loss
	accurac_val += 'Classifier Accuracy: %.3f%%; ' % (acc * 100)
	accurac_val += 'Discriminator Loss: Real: %.3f; Real_wrong: %.3f; Fake: %.3f; Fake_wrong: %.3f' % (d_loss_real, d_loss_real_wrong, d_loss_fake, d_loss_fake_wrong)
	# f = open('accuracy_record','a')
	# f.writelines(filename2+','+filename3+':'+accurac_val+'\n')
	# f.close()
	#print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
	# print('>Saved: %s, and %s' % (filename2, filename3))

	end = time.time()
	# f = open('time.txt','a')
	# f.writelines(str(step+1)+': '+str(end)+'.\n')
	# f.close()

# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=310, n_batch=32,
		  amt=100, num_train=100):
	total_time = 0

	# select supervised dataset
	X_sup, y_sup = select_supervised_samples(dataset)
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
	info_record = []
	prev_epoch = -1
	# manually enumerate epochs
	for i in range(n_steps):
		begin = time.time()
		# update supervised discriminator (c)
		[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)

		# c_model predicts the class
		c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
		# update unsupervised discriminator (d)
		[X_real, _], y_real = generate_real_samples(dataset, half_batch)
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		#X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		X_fake, y_fake = generate_fake_samples_grid(g_model, latent_dim, half_batch)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update generator (g)
		X_gan, y_gan = generate_latent_points_grid(latent_dim, n_batch), ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)

		total_time += (time.time() - begin)
		epoch = int((i+1) / bat_per_epo)

		print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
		if epoch % 50 == 0 and epoch > prev_epoch and epoch !=0:
			g_model.save(f'./h5/sgan_g_model_trainsize{amt}_epochs{epoch}.h5')
			c_model.save(f'./h5/sgan_c_model_trainsize{amt}_epochs{epoch}.h5')


	# all_times = np.array(TIMES)
	# # g_model.save(f'sgan_size{amt}.h5')
	# np.savetxt(f'sgan_trainsize{num_train}_times.txt', all_times)
	print(f'Time for {n_batch} batch size, {n_epochs} epochs, total time is {total_time}')



def main(types, num_train):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)

	# size of the latent space
	latent_dim = 1000
	# create the discriminator
	num_classes = len(types)
	d_model, c_model = define_discriminator(n_classes=num_classes)
	# create the generator
	g_model = define_generator(latent_dim)
	# create the gan
	gan_model = define_gan(g_model, d_model)
	# load image data
	dataset = load_real_samples_grid(num_types=num_classes, num_per_type=num_train)
	# train model
	train(g_model, d_model, c_model, gan_model, dataset, latent_dim,
			amt=num_train, num_train=num_train,
			n_epochs=2000)