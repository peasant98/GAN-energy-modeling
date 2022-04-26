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
from keras.models import load_model

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

# generate points in latent space as input for the generator
def generate_latent_points_grid(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(step, latent_dim, building_type=7, samples_to_gen=400, num_train=400, num_epochs=300):
    # load the model
    model_str = './h5/gan_type{building_type}_epochs{step}_trainsize{num_train}.h5'.format(building_type=building_type, step=step,
                                                                                           num_train=num_train)
    # model = load_model(f'./h5/gan_type{building_type}_epochs{step}_trainsize{num_train}.h5')
    model = load_model(model_str)

    zs = generate_latent_points_grid(latent_dim, samples_to_gen)
    prediction_str = './results/og_gan_results_trainsize{num_train}_epochs{step}_type_{building_type}.pickle'.format(num_train=num_train,
                                                                                                                     step=step,
                                                                                                                     building_type=building_type)
    prediction(zs, model, prediction_str, building_type=building_type)


def main(types, gens, num_train):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    #SELECTED_CLASSES = [7, 12, 14, 15]
    #SELECTED_CLASSES = [14, 15]
    
    #TIMES = []
    epochs = 2000
    latent_dim = 1000

    for idx, building_type in enumerate(types):
       i = 2000 
       summarize_performance(i, latent_dim, building_type=building_type, samples_to_gen=gens[idx], num_train=num_train[idx], num_epochs=epochs)
