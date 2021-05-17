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


def summarize_performance(step, latent_dim, num_classes=4,
						  num_train=400, gan_type='sgan',
                          epochs=10):

    """
    generate samples from sgan
    """    

    g_model_str = f'./h5/{gan_type}_g_model_trainsize{train_size}_epochs{epochs}.h5'

    g_model = load_model(g_model_str)

    c_model_str = f'./h5/{gan_type}_c_model_trainsize{train_size}_epochs{epochs}.h5'

    c_model = load_model(c_model_str)

    freq_dict = {7: 400, 12: 4500, 14: 800, 15: 400}

	for o in range(6):
		zs = generate_latent_points_grid(latent_dim, int(1600*3.14))
		# run some prediction
		prediction(zs, c_model, g_model, filename=f'./results/{gan_type}_results_trainsize{num_train}_epoch{epochs}_iter{o}.pickle')


def main(types, gens, num_train, gan_type='cgan'):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    latent_dim = 1000

    for i in range(50, 2001, 50):
        summarize_performance(i, latent_dim, num_classes=4, train_size=num_train, epochs=i,
							  gan_type=gan_type)
