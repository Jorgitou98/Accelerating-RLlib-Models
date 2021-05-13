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
        aggregated_data_this_model['mean_gpu_util_percent0'] = df['perf/gpu_util_percent0'].mean()
        aggregated_data_this_model['mean_vram_util_percent0'] = df['perf/vram_util_percent0'].mean()
        aggregated_data_this_model['mean_gpu_util_percent1'] = df['perf/gpu_util_percent1'].mean()
        aggregated_data_this_model['mean_vram_util_percent1'] = df['perf/vram_util_percent1'].mean()
        
        aggregated_results.append(aggregated_data_this_model)
    return aggregated_results
   

        
def get_data(directory, model_names, model_names_short, model, it_ini, it_fin):
    dir = dirname(dirname(abspath(__file__)))
    print(dir)
    aggregated_results = get_data_models(directory, model_names)
    
    vars = ['mean_gpu_util_percent0', 'mean_gpu_util_percent1', 'mean_vram_util_percent0', 'mean_vram_util_percent1']
    y_labels=['% util', '% util', '% util', '% util']
    
    for i in range(0,4,2):
        var_values_1 = [aggregated_results[j][vars[i]]*100 for j in range(0,len(model_names))]
        var_values_2 = [aggregated_results[j][vars[i+1]]*100 for j in range(0,len(model_names))]
        var_values=[var_values_1,var_values_2]
        title= vars[i] + ' and ' + vars[i+1] + ' model {}'.format(model)
        save_name = dir + '/result_analysis/training_results/volta1_def/graphs/' + vars[i] +'_and_' + vars[i+1] + '_model{}_it_'.format(model) + str(it_ini) + '_'+ str(it_fin) + '.png'
        plot_results.plot_bars_multiple(model_names_short, var_values, title, save_name, ['gpu0','gpu1'], y_label=y_labels[i])
    


def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    model = int(sys.argv[4])
    #model_names_str = sys.argv[5]
    #model_names = model_names_str[1:len(model_names_str)-1].split(',')
    model_names_short_str = sys.argv[5]
    model_names_short = model_names_short_str[1:len(model_names_short_str)-1].split(',')
    model_names_short = model_names_short_str[1:len(model_names_short_str)-1].split(',')
    model_names = ['model{}_{}'.format(model, i) for i in model_names_short]
    get_data(directory, model_names, model_names_short, model, it_ini, it_fin)

if __name__ == "__main__":
    main()        
