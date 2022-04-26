import numpy as np
import pandas as pd


if __name__ == '__main__':
    max_points = 400

    data = pd.read_csv('training_data/data_collect.csv')
    # 10k samples
    print(data)
    res = {}
    for val in data['label'].values:
        if val in res:
            res[val] += 1
        else:
            res[val] = 1
    new_df = pd.DataFrame()
    individual_dfs = []
    # label_set = [7, 12, 14, 15, 4, 5, 9, 10]
    label_set = [7, 12, 14, 15]

    amts = [40, 450, 80, 40, 20, 4, 25, 20, 10]

    idx = 0
    for i, label in enumerate(label_set):
        individual_df = pd.DataFrame()
        # rows = data[data['label'] == label][:amts[i]]
        rows = data[data['label'] == label][:max_points]
        rows['label'] = idx

        rows1 = data[data['label'] == label][:max_points]
        rows1['label'] = label
        idx += 1
        individual_df = individual_df.append(rows1)
        individual_dfs.append(individual_df)

        new_df = new_df.append(rows)

    print(new_df)

    # columns to take: 1 6 7 12 14 15 17
    # trim to 400.
    new_data = data[::10]
    # 1k samples
    # print(new_data)
    # res = {}
    # for val in new_data['label'].values:
    #     if val in res:
    #         res[val] += 1
    #     else:
    #         res[val] = 0
    # print(res)
    new_data.to_csv('./training_data/data_collect_select.csv', index=False)

    new_df.to_csv('./training_data/data_collect_select_equal.csv', index=False)

    for idx, val in enumerate(label_set):
        individual_dfs[idx].to_csv('./training_data/data_collect_select_class{val}.csv'.format(val=val), index=False)
