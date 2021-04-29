import csv
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from os.path import dirname, abspath
import plot_results



def get_data_models(directory, model_names):
    aggregated_results = []
    os.chdir(directory)
    for i in range (0, len(model_names)):        
        df = pd.read_csv(model_names[i] + '/progress.csv')
        df.dropna(inplace = True)
        aggregated_data_this_model = {}
        aggregated_data_this_model['mean_time_this_iter_s'] = df['time_this_iter_s'].mean()
        aggregated_data_this_model['mean_learn_time_ms'] = df['timers/learn_time_ms'].mean()
        aggregated_data_this_model['mean_sample_time_ms'] = df['timers/sample_time_ms'].mean()
        aggregated_data_this_model['mean_load_time_ms'] = df['timers/load_time_ms'].mean()
        aggregated_data_this_model['mean_update_time_ms'] = df['timers/update_time_ms'].mean()
        
        aggregated_results.append(aggregated_data_this_model)


    return aggregated_results
   

        
def get_data(directory, model_names, model_names_short, model, it_ini, it_fin):
    dir = dirname(dirname(abspath(__file__)))
    print(dir)
    aggregated_results = get_data_models(directory, model_names)
    
    vars = ['mean_sample_time_ms','mean_load_time_ms','mean_learn_time_ms','mean_update_time_ms']
    
    var_values = []
    for i in range(len(vars)):
        var_values.append([aggregated_results[j][vars[i]] for j in range(len(model_names))])
    print(var_values)
    title= 'Times distribution model {}'.format(model)
    save_name = dir + '/result_analysis/training_results/volta1_def/graphs/' + 'times_distribution_model{}_it_'.format(model) + str(it_ini) + '_'+ str(it_fin) + '.png'
    plot_results.plot_bars_stacked(model_names_short, var_values, title, save_name, labels = vars, y_label='time(ms)')

def get_data_percent(directory, model_names, model_names_short, model, it_ini, it_fin):
    dir = dirname(dirname(abspath(__file__)))
    print(dir)
    aggregated_results = get_data_models(directory, model_names)
    
    vars = ['mean_sample_time_ms','mean_load_time_ms','mean_learn_time_ms','mean_update_time_ms']
    
    var_values = []
    for i in range(len(vars)):
        var_values.append([aggregated_results[j][vars[i]] for j in range(len(model_names))])
    print(var_values)
    for i in range(len(model_names)):
        times_sum = sum([var_values[k][i] for k in range(len(vars))])
        print(times_sum)
        for j in range(len(vars)):
            var_values[j][i] = (var_values[j][i]/times_sum) * 100

    print(var_values)
    title= 'Times distribution percent model {}'.format(model)
    save_name = dir + '/result_analysis/training_results/volta1_def/graphs/' + 'times_distribution_percent_model{}_it_'.format(model) + str(it_ini) + '_'+ str(it_fin) + '.png'
    plot_results.plot_bars_stacked(model_names_short, var_values, title, save_name, labels = vars, y_label='total iter time %')
    


def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    model = int(sys.argv[4])

    '''
    model_names_str = sys.argv[5]
    model_names = model_names_str[1:len(model_names_str)-1].split(',')
    model_names_short_str = sys.argv[6]
    model_names_short = model_names_short_str[1:len(model_names_short_str)-1].split(',')
    '''
    model_options_str = sys.argv[5]
    model_options = model_options_str[1:len(model_options_str)-1].split(',')
    model_names_short = []
    for gpu in ['gpu0', 'gpu1', 'both_gpus']:
        for option in model_options:
            model_names_short.append('{}_{}'.format(gpu, option))
    model_names = ['model{}_{}'.format(model, i) for i in model_names_short]

    get_data_percent(directory, model_names, model_names_short, model, it_ini, it_fin)  

if __name__ == "__main__":
    main()        
