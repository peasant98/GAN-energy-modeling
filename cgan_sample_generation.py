from numpy import zeros
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


from tensorflow import ConfigProto
from tensorflow import InteractiveSession

SELECTED_CLASSES = [7, 12, 14, 15, 9, 5]
# SELECTED_CLASSES = [4, 5, 9, 10]

TIMES = []


def denormalize(power_predictions, class_predictions, csv_path='data_collect_maxmin.csv', filename='cgan_results.pickle'):
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

	final = denormalize(power_predictions, class_predictions, filename=filename)


# generate points in latent space as input for the generator
def generate_latent_points_grid(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y


def summarize_performance(step, latent_dim, num_classes=4,
						  train_size=100,
                          epochs=10,
						  freq_dict={7: 400, 12: 4500, 14: 800, 15: 400},
						  gan_type='cgan'):
    model_str = f'./h5/{gan_type}_trainsize{train_size}_epochs{epochs}.h5'

    g_model = load_model(model_str)

    num_gen = 0
    for val in freq_dict:
        num_gen += (freq_dict[val])
    zs, _ = generate_latent_points(latent_dim, num_gen, n_classes=num_classes)
    # run some prediction
	# compute classes
    classes = []

    idx = 0
    for val in freq_dict:
        print(val, freq_dict[val], idx)
        for _ in range(freq_dict[val]):
            classes.append(idx)
        idx += 1

    # run prediction
    prediction(zs, np.array(classes), g_model, filename=f'./results/{gan_type}_results_trainsize{train_size}_epoch{epochs}.pickle')


def main(types, gens, num_train, gan_type='cgan'):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)

	latent_dim = 1000
	freq_dict = {}
	for idx, val in enumerate(types):
		freq_dict[val] = gens[idx]

	n_classes = len(types)
	for i in range(50, 2001, 50):
		summarize_performance(i, latent_dim, num_classes=n_classes, train_size=num_train, epochs=i,
								gan_type=gan_type, freq_dict=freq_dict)
