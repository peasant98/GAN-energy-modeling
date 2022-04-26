# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack, asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

import pickle
import pandas as pd
import numpy as np
import time

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

def denormalize(power_predictions, building_type, csv_path='./training_data/data_collect_maxmin.csv', filename='og_gan_results.pickle'):
	df = pd.read_csv(csv_path)
	final_arr = []
	# results is a list of lists -- each entry contains building_type id, followed
	# by the normalized val.
	for idx, power_prediction in enumerate(power_predictions):
		row = df.values[building_type]
		max_val = row[1]
		min_val = row[2]
		denormalized_val = power_prediction * (max_val - min_val) + (max_val + min_val)
		denormalized_val /= 2
		final_arr.append([building_type, denormalized_val])

	# convert to pickle file for now.
	# print(final_arr[0][1].shape)
	with open(filename, 'wb') as handle:
		pickle.dump(final_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return final_arr


def prediction(zs, g_model, filename, building_type=7):
	# predict normalized value of generator with latent space as input.
	results = []
	normalized_val = g_model.predict(zs)
	# predict class with normalized val
	power_predictions = normalized_val.reshape(normalized_val.shape[0], normalized_val.shape[1])
	denormalize(power_predictions, building_type=building_type, filename=filename)



def define_discriminator(in_shape=(744, 1, 1)):
    # define the standalone discriminator model
	model = Sequential()
	model.add(Conv2D(128, (3,3), strides=(2, 1), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2, 1), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (3,3), strides=(2, 1), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 23808
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((186, 1, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,1), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,1), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def load_real_samples_grid(class_num=7, num_train=400):
	# load the REAL data.
	f = open('./training_data/data_collect_select_class{}.csv'.format(class_num), 'r')
	# f = open(f'./data_collect_select_class{class_num}.csv','r')
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
	X = expand_dims(X1, axis=-1)[:num_train]
	# print(X.shape, trainy.shape)
	print(X.shape)
	return X



def select_supervised_samples(dataset, n_samples=1000, n_classes=18):
	X, y = dataset
	return asarray(X), asarray(y)


# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images and labels
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y


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
	# print(z_input.shape)
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=310, n_batch=64,
		  building_type=7, samples_to_gen=400,
		  num_train=400):
	total_time = 0
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		begin = time.time()
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples_grid(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points_grid(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			# print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes

		total_time += (time.time() - begin)
		print('>%d, d=%.3f, g=%.3f' % (i+1, d_loss, g_loss))
		if (i+1) % 50 == 0:
			# every 10 epochs
			# TIMES.append(total_time)
			# saves the model
			# g_model.save(f'./h5/gan_type{building_type}_epochs{i+1}_trainsize{num_train}.h5')
			g_model_filename = './h5/gan_type{building_type}_epochs{j}_trainsize{num_train}.h5'.format(building_type=building_type, j=i+1, num_train=num_train)
			g_model.save(g_model_filename)

	#print(f'Time for {n_batch} batch size, {n_epochs} epochs, total time is {total_time}')
	#all_times = np.array(TIMES)
	#np.savetxt(f'./h5/gan_type{building_type}_epoch_{n_epochs}_num_train{num_train}_times.txt', all_times)


#if __name__ == '__main__':
def main(types, gens, num_train):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    #SELECTED_CLASSES = [7, 12, 14, 15]
    #SELECTED_CLASSES = [14, 15]
    
    # TIMES = []

    # size of the latent space
    latent_dim = 1000
    epochs = 2000

    for idx, building_type in enumerate(types):
        # building_type = 1
        samples_to_gen = gens[idx]
        # create the discriminator
        d_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # load image data
        dataset = load_real_samples_grid(class_num=building_type, num_train=num_train[idx])
        # train model
        n_batch = 64
        if num_train[idx] < 32:
            n_batch = int(num_train[idx] / 2)
        elif num_train[idx] < 64:
            n_batch = 16
        elif num_train[idx] < 128:
            n_batch = 32
        train(g_model, d_model, gan_model, dataset, latent_dim,building_type=building_type,samples_to_gen=samples_to_gen,n_batch=n_batch, num_train=num_train[idx],n_epochs=epochs)
