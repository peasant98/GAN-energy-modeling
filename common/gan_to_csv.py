import pickle

import pandas as pd
import numpy as np


def sample_load_results(filename):
    generated_results = pickle.load(open(filename, 'rb'))
    """
    generated results is formatted like so:

    it is an array of size n, where item in the array is as follows:

    [building_type, power values_arr], where power_values_arr is of size 744,
    indicating the power usage over a certain timeframe
    """
    # a code example
    first_entry = generated_results[0]
    building_label = first_entry[0]
    power_predictions = first_entry[1]
    # print(power_predictions)


def convert_results_pickle_to_csv(filename, outfilename):
    generated_results = pickle.load(open(filename, 'rb'))
    data = pd.read_csv('./training_data/data_collect.csv')
    new_data = pd.DataFrame(columns=data.columns[1:])
    rows = []
    x = []
    total = {}
    for i, gen_result in enumerate(generated_results):
        data_dict = {}
        print(i)
        for idx, pred in enumerate(gen_result[1]):
            data_dict[new_data.columns[idx]] = pred
        data_dict['label'] = gen_result[0]
        x.append(gen_result[0])
        if gen_result[0] in total:
            total[gen_result[0]] += 1
        else:
            total[gen_result[0]] = 1

        rows.append(data_dict)
    print(total)
    new_data = new_data.append(rows)
    new_data.to_csv(outfilename, index=False)


def combine_pickle_data(filenames_list, outfilename):
    """
    combines list of pickle data
    """
    data = pd.read_csv('./training_data/data_collect.csv')
    # hard-coded for now
    desired_amt = {7: 400,
                   12: 4500,
                   14: 800,
                   15: 400}
    main_amt = {7: 400,
                12: 4500,
                14: 800,
                15: 400}
    new_data = pd.DataFrame(columns=data.columns[1:])
    rows = []
    x = []
    total = {}
    done = False
    for filename in filenames_list:
        if done:
            break
        generated_results = pickle.load(open(filename, 'rb'))

        for i, gen_result in enumerate(generated_results):
            data_dict = {}
            print(i)
            for idx, pred in enumerate(gen_result[1]):
                data_dict[new_data.columns[idx]] = pred

            data_dict['label'] = gen_result[0]
            x.append(gen_result[0])
            if gen_result[0] in total:
                if total[gen_result[0]] + 1 > main_amt[gen_result[0]]:
                    continue
                else:
                    total[gen_result[0]] += 1
            else:
                total[gen_result[0]] = 1
            desired_amt[gen_result[0]] -= 1

            rows.append(data_dict)

            empty_total = 0
            for val in desired_amt:
                if desired_amt[val] <= 0:
                    empty_total += 1

            if empty_total == len(desired_amt.keys()):
                done = True
                break

        print(total)
    new_data = new_data.append(rows)
    new_data.to_csv(outfilename, index=False)


#if __name__ == '__main__':
def main(types, num_train):
    # load the results
    # sample_load_results('gan_results2575.pickle')
    for building_type in types:
        for train_size in num_train:
            for i in [2000]:
                # basefilename = f'./results/og_gan_results_trainsize{train_size}'

                basefilename = './results/og_gan_results_trainsize{train_size}'.format(train_size=train_size)

                # pickle_name = f'{basefilename}_epochs{i}_type_{building_type}.pickle'
                pickle_name = '{basefilename}_epochs{i}_type_{building_type}.pickle'.format(basefilename=basefilename,
                                                                                            i=i,
                                                                                            building_type=building_type)

                # outcsvname = f'{basefilename}_epochs{i}_type_{building_type}.csv'
                outcsvname = '{basefilename}_epochs{i}_type_{building_type}.csv'.format(basefilename=basefilename,
                                                                                        i=i,
                                                                                        building_type=building_type)

                convert_results_pickle_to_csv(pickle_name,outfilename=outcsvname)

