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

SELECTED_CLASSES = [7, 12, 14, 15]
# SELECTED_CLASSES = [4, 5, 9, 10]

TIMES = []


def denormalize(power_predictions, class_predictions, csv_path='./training_data/data_collect_maxmin.csv'):
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


	return final_arr


def prediction(zs, c_model, g_model):
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
	final = denormalize(power_predictions, class_predictions)
	return final

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

def summarize_performancev2(step, latent_dim, num_classes=4,
						  num_train=400, train_size=100, gan_type='sgan',
                          epochs=10, gen_dict=None):

	"""
	generate samples from sgan
	"""    

	g_model_str = f'./h5/{gan_type}_g_model_trainsize{train_size}_epochs{epochs}.h5'

	g_model = load_model(g_model_str)

	c_model_str = f'./h5/{gan_type}_c_model_trainsize{train_size}_epochs{epochs}.h5'

	c_model = load_model(c_model_str)

	all_predictions = []
	filename = f'./results/{gan_type}_results_trainsize{train_size}_epoch{epochs}.pickle'

	for _ in range(10):

		zs = generate_latent_points_grid(latent_dim, 610)
		# run some prediction
		predictions = prediction(zs, c_model, g_model)
		for entry in predictions:
			building_type = entry[0]
			all_predictions.append(entry)
	

	with open(filename, 'wb') as handle:
		pickle.dump(all_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def finished_generation(desired_dict, current_dict):
	for key in desired_dict:
		if desired_dict[key] != current_dict[key]:
			return False
	
	return True


def summarize_performance(step, latent_dim, num_classes=4,
						  num_train=400, train_size=100, gan_type='sgan',
                          epochs=10, gen_dict=None):

	"""
	generate samples from sgan
	"""    

	g_model_str = f'./h5/{gan_type}_g_model_trainsize{train_size}_epochs{epochs}.h5'

	g_model = load_model(g_model_str)

	c_model_str = f'./h5/{gan_type}_c_model_trainsize{train_size}_epochs{epochs}.h5'

	c_model = load_model(c_model_str)

	if gen_dict is None:
		freq_dict = {7: 400, 12: 4500, 14: 800, 15: 400}
	else:
		freq_dict = gen_dict

	all_predictions = []
	filename = f'./results/{gan_type}_results_trainsize{train_size}_epoch{epochs}.pickle'

	if gen_dict is None:
		current_freq_dict = {7: 0, 12: 0, 14: 0, 15: 0}
	else:
		current_freq_dict = {}
		for key in gen_dict:
			current_freq_dict[key] = 0

	for _ in range(34):

		if finished_generation(freq_dict, current_freq_dict):
			break

		zs = generate_latent_points_grid(latent_dim, 1600)
		# run some prediction
		predictions = prediction(zs, c_model, g_model)
		for entry in predictions:
			building_type = entry[0]

			if current_freq_dict[building_type] < freq_dict[building_type]:
				all_predictions.append(entry)
				current_freq_dict[building_type] += 1
	

	with open(filename, 'wb') as handle:
		pickle.dump(all_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def finished_generation(desired_dict, current_dict):
	for key in desired_dict:
		if desired_dict[key] != current_dict[key]:
			return False
	
	return True


def main(types, gens, num_train, gan_type='sgan'):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	# construct types to amount to generate dictionary

	gen_dict = {}
	for idx in range(len(types)):
		gen_dict[types[idx]] = gens[idx]


	latent_dim = 1000

	for i in range(1900, 2001, 50):
		summarize_performancev2(i, latent_dim, num_classes=4, train_size=num_train, epochs=i,
								gan_type=gan_type, gen_dict=gen_dict, num_train=num_train)
