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



def get_data_models(directory, model_names, model_names_short):
    aggregated_results = []
    os.chdir(directory)
    for i in range (0, len(model_names)):        
        df = pd.read_csv(model_names[i] + '/progress.csv')
        df.dropna(inplace = True)
        aggregated_data_this_model = {}
        aggregated_data_this_model['model'] = model_names_short[i]
        
        '''
        aggregated_data_this_model['mean_time_this_iter_s'] = df['time_this_iter_s'].mean()
        aggregated_data_this_model['mean_learn_time_ms'] = df['timers/learn_time_ms'].mean()
        aggregated_data_this_model['mean_sample_time_ms'] = df['timers/sample_time_ms'].mean()
        aggregated_data_this_model['mean_load_time_ms'] = df['timers/load_time_ms'].mean()
        aggregated_data_this_model['mean_update_time_ms'] = df['timers/update_time_ms'].mean()
        '''

        
        aggregated_data_this_model['mean_cpu_util_percent'] = df['perf/cpu_util_percent'].mean()
        aggregated_data_this_model['mean_ram_util_percent'] = df['perf/ram_util_percent'].mean()
        
        aggregated_results.append(aggregated_data_this_model)

    
    return aggregated_results
        
def get_data(directory, model_names_short, model_ids, it_ini, it_fin):
    dir = dirname(dirname(abspath(__file__)))
    model_names = [['model{}_{}'.format(model_ids[j], model_names_short[i]) for i in range(0, len(model_names_short))] for j in range(0, len(model_ids))]
    aggregated_results_list = [get_data_models(directory, model_names[i], model_names_short) for i in range(0, len(model_ids))]
    
    #vars = ['mean_time_this_iter_s','mean_sample_time_ms','mean_load_time_ms','mean_learn_time_ms','mean_update_time_ms']
    #y_labels=['time(s)', 'time(ms)', 'time(ms)', 'time(ms)', 'time(ms)']
    vars = ['mean_cpu_util_percent', 'mean_ram_util_percent']
    y_labels = ['% util', '% util']
    
    for i in range(len(vars)):
        var_values = [[aggregated_results_list[k][j][vars[i]] for j in range(0,len(model_names_short))] for k in range(0, len(model_ids))]
        title= 'Compare ' + vars[i] + ' all models'
        save_name = dir + '/result_analysis/training_results/volta1_def/graphs/compare_' + vars[i] + '_all_models_it_' + str(it_ini) + '_'+ str(it_fin) + '.png'
        plot_results.plot_bars_multiple(model_names_short, var_values, title, save_name, labels = ['model1', 'model3', 'model4'], y_label=y_labels[i])

def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    model_ids_str = sys.argv[4]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    model_options_str = sys.argv[5]
    model_options= model_options_str[1:len(model_options_str)-1].split(',')
    model_names_short=[]
    for gpu in ['gpu0', 'gpu1', 'both_gpus']:
        for option in model_options:
            model_names_short.append('{}_{}'.format(gpu, option))
    get_data(directory, model_names_short, model_ids, it_ini, it_fin)        

if __name__ == "__main__":
    main()        
