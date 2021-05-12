# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
from keras.initializers import RandomNormal
from numpy import zeros
from numpy import ones, asarray
from numpy import expand_dims
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
from keras.layers import Concatenate, BatchNormalization

import pandas as pd
import pickle
import numpy as np
import time


from tensorflow import ConfigProto
from tensorflow import InteractiveSession

import pandas as pd

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# SELECTED_CLASSES = [7, 12, 14, 15, 4, 5, 9, 10]
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
def define_discriminator(in_shape=(744,1,1), n_classes=4):
	# weight initialization
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# # normal
	fe = Conv2D(64, (3,3), padding='same')(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# # downsample to 7x7
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# # normal
	fe = Conv2D(256, (3,3), padding='same')(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)

	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# # flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)


	# real/fake output
	out1 = Dense(1, activation='sigmoid')(fe)
	# class label output
	out2 = Dense(n_classes, activation='softmax')(fe)
	# define model
	model = Model(in_image, [out1, out2])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=4):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 186
	li = Dense(n_nodes, kernel_initializer=init)(li)
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
	# connect the outputs of the generator to the inputs of the discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and label and outputting real/fake and label outputs
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
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
def generate_latent_points(latent_dim, n_samples, n_classes=4):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# generate samples and save as a plot and save the model
"""
freq_dict={7: 400, 12: 4500, 14: 800, 15:400,
						  			 4: 40, 5: 250, 9: 200, 10: 100}
"""
def summarize_performance(step, g_model, latent_dim, n_samples=400,
                          freq_dict={7: 400, 12: 4500, 14: 800, 15:400},
                          num_classes=4,
						  amt=100):
	# prepare fake examples

    num_gen = 0
    for val in freq_dict:
        num_gen += (freq_dict[val])
    print(num_gen, ' will be generated')
    zs, _ = generate_latent_points(latent_dim, num_gen, n_classes=num_classes)

    classes = []

    idx = 0
    for val in freq_dict:
        print(val, freq_dict[val], idx)
        for j in range(freq_dict[val]):
            classes.append(idx)
        idx += 1

    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)

    # run prediction
    prediction(zs, np.array(classes), g_model, filename=f'acgan_results_trainsize{amt}_epoch{step}.pickle')


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=310, n_batch=32,
          amt=100, num_classes=4):
    total_time = 0
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
		# enumerate batches over the training set
        begin = time.time()

        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
            # generate 'fake' examples
            [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
            # prepare points in latent space as input for the generator
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
            # summarize loss on this batch
            # evaluate the model performance every 'epoch'
            total_time += (time.time() - begin)

        if (i+1) % 50 == 0:
            g_model.save(f'./h5/acgan_trainsize{amt}_epochs{i+1}.h5')
            # summarize_performance(i, g_model, latent_dim,
			# 					     amt=amt)

        print('epoch >%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
    # save the generator model
    # all_times = np.array(TIMES)
	# # g_model.save(f'cgan_size{amt}.h5')
    # np.savetxt(f'acgan_trainsize{amt}_times.txt', all_times)



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
