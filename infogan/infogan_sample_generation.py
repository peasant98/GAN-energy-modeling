from numpy import zeros, hstack
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from tensorflow.keras.utils import to_categorical


import pickle
import pandas as pd
import numpy as np
import itertools

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession

SELECTED_CLASSES = [7, 12, 14, 15]
# SELECTED_CLASSES = [4, 5, 9, 10]

TIMES = []


def denormalize(power_predictions, class_predictions, cat_codes, csv_path='./training_data/data_collect_maxmin.csv', filename='gan_results.pickle'):
	df = pd.read_csv(csv_path)
	# results is a list of lists -- each entry contains building_type id, followed
	# by the normalized val.
	final_arr = []


	for idx, power_prediction in enumerate(power_predictions):
		building_type = class_predictions[idx]
		row = df.values[SELECTED_CLASSES[building_type]]
		max_val = row[1]
		min_val = row[2]
		denormalized_val = power_prediction * (max_val - min_val) + (max_val + min_val)
		denormalized_val /= 2
		final_arr.append([SELECTED_CLASSES[building_type], cat_codes[idx], denormalized_val])

	# convert to pickle file for now.
	with open(f'./results/{filename}', 'wb') as handle:
		pickle.dump(final_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return final_arr


def prediction(zs, classes, g_model, c_model, filename):
	# predict normalized value of generator with latent space as input.
	results = []
	normalized_val = g_model.predict(zs)
	# predict class with normalized val
	for i in range(50, 2001, 50):

		filename = f'infogan_results_trainsize100_epoch{i}.pickle'

		c_model_str = f'./h5/sgan_c_model_trainsize100_epochs{i}.h5'
		c_model = load_model(c_model_str)
		building_type_id = c_model.predict(normalized_val)
		class_predictions = np.argmax(building_type_id, axis=-1)
		print(i)	
		print(np.unique(class_predictions))

		power_predictions = normalized_val.reshape(normalized_val.shape[0], normalized_val.shape[1])
		preds = {}
		for prediction in class_predictions:
			if prediction in preds:
				preds[prediction] += 1
			else:
				preds[prediction] = 1
		print(np.unique(class_predictions))
		print(preds)
		final = denormalize(power_predictions, class_predictions, classes, filename=filename)



# generate points in latent space as input for the generator
def generate_latent_points_grid(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_cat, n_samples,
						   return_classes=False,
						   freq_dict={7: 400, 12: 4500, 14: 800, 15: 400},
						   generate_eval=False,
						   keys=[7,12,14,15]):

	# generate points in the latent space
	if freq_dict is None:
		freq_dict = {7: 4500, 12: 4500, 14: 4500, 15: 4500}

	if generate_eval:
		total_sum = 0
		for val in freq_dict:
			total_sum += (freq_dict[val])

		z_latent = randn(latent_dim * total_sum)
		# reshape into a batch of inputs for the network
		z_latent = z_latent.reshape(total_sum, latent_dim)
		# generate categorical codes
		cat_codes = []
		cur_code = 0

		for key in keys:
			for _ in range(freq_dict[key]):
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


def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y


def summarize_performance(step, latent_dim, n_cat, train_size=100, epochs=0, n_samples=1600,
						  gen_dict=None, keys=[7,12,14,15]):

	# generate some samples
	g_model_str = f'./h5/infogan_trainsize{train_size}_epochs{epochs}.h5'

	g_model = load_model(g_model_str)

	c_model_str = f'./h5/sgan_c_model_trainsize{train_size}_epochs1500.h5'

	c_model = load_model(c_model_str)

	# create multiple files for different ordering of control codes
	keys_permutations = list(itertools.permutations(keys))
	keys_permutations = [keys]
	for perm in keys_permutations:
		list_perm = list(perm)
		[z, _], classes = generate_latent_points(latent_dim, n_cat, n_samples,
												return_classes=True,
												generate_eval=True, freq_dict=gen_dict, keys=list_perm)
					
		prediction(z, classes, g_model, c_model, f'infogan_results_trainsize{train_size}_epoch{epochs}.pickle')


def main(types, gens, num_train):
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)


	gen_dict = {}
	for idx in range(len(types)):
		gen_dict[types[idx]] = gens[idx]

	n_cat = len(types)
	latent_dim = 1000

	for i in range(2000, 2001, 50):
		summarize_performance(i, latent_dim, n_cat=n_cat, train_size=num_train, epochs=i,
							  gen_dict=gen_dict, keys=types)
