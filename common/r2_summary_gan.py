import pandas as pd
import numpy as np

def collect_data(path_csv):
    data = pd.read_csv(path_csv)
    return data

def r_squared(generated_list,initial_list):
    y_bar = np.average(initial_list)
    ss_tot = 0
    ss_res = 0
    for i in range(len(initial_list)):
        ss_tot += (initial_list[i]-y_bar)**2
        ss_res += (initial_list[i]-generated_list[i])**2
    r_2 = 1 - ss_res/ss_tot
    return r_2

def day_hour_list():
    day_hour = []
    for i in range(1,32):
        for j in range(1,25):
            if i < 10:
                day_hour.append('08/0'+str(i)+'_'+str(j))
            else:
                day_hour.append('08/'+str(i)+'_'+str(j))
    return day_hour

def key_parameter_building_type(data):
    day_hour = day_hour_list()
    data_building_type = []
    for label in range(18):
        temp = []
        for i in range(len(data)):
            if data['label'][i] == label:
                temp_1 = []
                for x in day_hour:
                    temp_1.append(data[x][i])
                temp.append(temp_1)
        data_building_type.append(temp)
    avg = []
    median = []
    low_percent = []
    high_percent = []
    iqr = []
    for i in range(18):
        avg_temp = []
        median_temp = []
        low_percent_temp = []
        high_percent_temp = []
        iqr_temp = []
        if len(data_building_type[i]) > 0:
            for j in range(len(day_hour)):
                temp = []
                for row in data_building_type[i]:
                    temp.append(row[j])
                avg_temp.append(np.average(temp))
                median_temp.append(np.median(temp))
                low_percent_temp.append(np.percentile(temp,25))
                high_percent_temp.append(np.percentile(temp,75))
                iqr_temp.append(np.percentile(temp,75)-np.percentile(temp,25))
        avg.append(avg_temp)
        median.append(median_temp)
        low_percent.append(low_percent_temp)
        high_percent.append(high_percent_temp)
        iqr.append(iqr_temp)
    return avg,median,low_percent,high_percent,iqr

def key_parameter_all(data,label_list):
    day_hour = day_hour_list()
    avg = []
    median = []
    low_percent = []
    high_percent = []
    iqr = []
    for x in day_hour:
        temp = []
        for i in range(len(data)):
            if data['label'][i] in label_list:
                temp.append(data[x][i])
        avg.append(np.average(temp))
        median.append(np.median(temp))
        low_percent.append(np.percentile(temp,25))
        high_percent.append(np.percentile(temp,75))
        iqr.append(np.percentile(temp,75)-np.percentile(temp,25))
    return avg,median,low_percent,high_percent,iqr

def calculate(path_csv_list,gan_type,train_size,epochs,label_list):
    initial = collect_data('./training_data/power_new.csv')
    for ind,path_csv in enumerate(path_csv_list):
        if ind == 0:
            generated = collect_data(path_csv)
        else:
            temp = collect_data(path_csv)
            generated = pd.concat([generated,temp],ignore_index = True)
    summary = []
    # based on building type
    avg_init,median_init,low_percent_init,high_percent_init,iqr_init = key_parameter_building_type(initial)
    avg_gen,median_gen,low_percent_gen,high_percent_gen,iqr_gen = key_parameter_building_type(generated)
    for i in range(18):
        if len(avg_gen[i]) > 0:
            temp = [gan_type,train_size,epochs,'building_type_'+str(i)]
            temp.append(r_squared(avg_gen[i],avg_init[i]))
            temp.append(r_squared(median_gen[i],median_init[i]))
            temp.append(r_squared(low_percent_gen[i],low_percent_init[i]))
            temp.append(r_squared(high_percent_gen[i],high_percent_init[i]))
            temp.append(r_squared(iqr_gen[i],iqr_init[i]))
            summary.append(temp)
    # all
    avg_init,median_init,low_percent_init,high_percent_init,iqr_init = key_parameter_all(initial,label_list)
    avg_gen,median_gen,low_percent_gen,high_percent_gen,iqr_gen = key_parameter_all(generated,label_list)
    temp = [gan_type,train_size,epochs,'building_type_all']
    temp.append(r_squared(avg_gen,avg_init))
    temp.append(r_squared(median_gen,median_init))
    temp.append(r_squared(low_percent_gen,low_percent_init))
    temp.append(r_squared(high_percent_gen,high_percent_init))
    temp.append(r_squared(iqr_gen,iqr_init))
    summary.append(temp)
    return summary

def main(types, num_train, i):
    for gan_type in ['og_gan']:
        for train_size in num_train:
            final_table = []
            for epochs in range(50,2001,50):
                path_csv_list = []
                for building_type in types:
                    path_csv_list.append('./results/'+gan_type+'_results_trainsize'+str(train_size)+'_epochs'+str(epochs)+'_type_'+str(building_type)+'.csv')
                summary = calculate(path_csv_list,gan_type,train_size,epochs,types)
                for row in summary:
                    final_table.append(row)
            df = pd.DataFrame(final_table,columns = ['gan_type','train_size','epochs','building_type','avg_r2','median_r2','25th_percent_r2','75th_percent_r2','iqr_r2'])
            df.to_csv('./'+str(i)+'_csv/1_'+gan_type+'_trainsize'+str(train_size)+'.csv',index = False)
