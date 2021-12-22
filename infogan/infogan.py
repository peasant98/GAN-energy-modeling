# example of training an infogan
from keras.layers.core import Dropout
import numpy as np
from numpy import zeros, asarray, ones, expand_dims, hstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
    Activation,
	Dropout
)
from matplotlib import pyplot
import pandas as pd
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

# denormalize the predictions.
def denormalize(power_predictions, class_predictions, csv_path='../data/data_collect_maxmin.csv', filename='gan_results.pickle'):
    df = pd.read_csv(csv_path)
    final_arr = []
    # results is a list of lists -- each entry contains building_type id, followed
    # by the normalized val.
    for idx, power_prediction in enumerate(power_predictions):
        building_type = int(class_predictions[idx])
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
	normalized_val = g_model.predict(zs)
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
def define_discriminator(n_cat, in_shape=(744,1,1)):
	# weight initialization
	# image input
	in_image = Input(shape=in_shape)


	fe = Conv2D(128, (3,3), strides=(2, 1), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Conv2D(128, (3,3), strides=(2, 1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	
	fe = Conv2D(128, (3,3), strides=(2, 1), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Flatten()(fe)
	fe = Dropout(0.4)(fe)


	# real/fake output, probability of being fake or not
	out_classifier = Dense(1, activation='sigmoid')(fe)
	# define d model
	d_model = Model(in_image, out_classifier)
	# compile discriminator model
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# create q model layers
	q = Dense(128)(fe)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.2)(q)
	# q model output
	out_codes = Dense(n_cat, activation='softmax')(q)
	# define q model
	q_model = Model(in_image, out_codes)
	return d_model, q_model


# define the standalone generator model
def define_generator(gen_input_size):

	# image generator input
	in_lat = Input(shape=(gen_input_size,))
	# foundation for 7x7 image
	n_nodes = 128 * 186
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
	# out_layer = Reshape((744, 1, 1))(out_layer)

	# define model
	model = Model(in_lat, out_layer)
	return model

# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model, q_model):
	# make weights in the discriminator (some shared with the q model) as not trainable
	d_model.trainable = False
	# connect g outputs to d inputs
	d_output = d_model(g_model.output)
	# connect g outputs to q inputs
	q_output = q_model(g_model.output)
	# define composite model
	model = Model(g_model.input, [d_output, q_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
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

	return [np.array(new_X), np.array(new_trainy)]

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset[0].shape[0], n_samples)
	# select images and labels
	X = dataset[0][ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_cat, n_samples,
						   return_classes=False,
						   freq_dict={7: 400, 12: 4500, 14: 800, 15: 400},
						   generate_eval=False):
	# generate points in the latent space
	if generate_eval:
		total_sum = 0
		for val in freq_dict:
			total_sum += (freq_dict[val])

		z_latent = randn(latent_dim * total_sum)
		# reshape into a batch of inputs for the network
		z_latent = z_latent.reshape(total_sum, latent_dim)
		# generate categorical codes
		cat_codes = randint(0, n_cat, n_samples)
		cat_codes = []
		cur_code = 0
		for val in freq_dict:
			for _ in range(freq_dict[val]):
				cat_codes.append(cur_code)
			cur_code += 1
		cat_codes = np.array(cat_codes)
	else:
		z_latent = randn(latent_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_latent = z_latent.reshape(n_samples, latent_dim)
		# generate categorical codes
		cat_codes = randint(0, n_cat, n_samples)

	list_selected_classes = cat_codes
	# one hot encode
	cat_codes = to_categorical(cat_codes, num_classes=n_cat)
	# concatenate latent points and control codes
	z_input = hstack((z_latent, cat_codes))
	if return_classes:
		return [z_input, cat_codes], list_selected_classes
	return [z_input, cat_codes]

def generate_latent_points_specific_class(latent_dim, n_cat, n_samples, building_type):
    # generate points in the latent space
	z_latent = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_latent = z_latent.reshape(n_samples, latent_dim)
	# define categorical codes
	cat_codes = asarray([building_type for _ in range(n_samples)])
	# one hot encode
	cat_codes = to_categorical(cat_codes, num_classes=n_cat)
	# concatenate latent points and control codes
	z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_cat, n_samples):
	# generate points in latent space and control codes
	z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, gan_model, latent_dim, n_cat, n_samples=1600,
						  trainsize=400):

	# generate some samples
	[z, _], classes = generate_latent_points(latent_dim, n_cat, n_samples,
								  			 return_classes=True,
											   generate_eval=True)
	prediction(z, classes, g_model, f'info_gan_trainsize{trainsize}_epochs{step}.pickle')


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_cat, n_epochs=300, n_batch=32,
		  amt=400):
    total_time = 0

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):

        begin = time.time()

        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator and q model weights
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, half_batch)
            # update discriminator model weights
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the g via the d and q error
            _,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
            # summarize loss on this batch
        print('>%d, d[%.3f,%.3f], g[%.3f] q[%.3f]' % (i+1, d_loss1, d_loss2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        total_time += (time.time() - begin)

        if (i+1) % 50 == 0:
            # summarize_performance(i, g_model, gan_model, latent_dim, n_cat,
            #                         trainsize=amt)
            g_model.save(f'./h5/infogan_trainsize{amt}_epochs{i+1}.h5')



def main(types, num_train):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # size of the latent space
    latent_dim = 1000
    # create the discriminator
    n_cat = len(types)
    d_model, q_model = define_discriminator(n_cat)
    # create the generator
    gen_input_size = latent_dim + n_cat
    g_model = define_generator(gen_input_size)
    # create the gan
    gan_model = define_gan(g_model, d_model, q_model)
    # load image data
    dataset = load_real_samples_grid(num_types=n_cat, num_per_type=num_train)
    # train model
    train(g_model, d_model, gan_model, dataset, latent_dim,
          n_cat, amt=num_train,
          n_epochs=2000)
