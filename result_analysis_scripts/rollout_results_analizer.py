
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

def get_data_models(directory, n_iters, gpu, save_directory):
    if gpu:
        name = 'model{}_gpu_{}_iters.csv'
        aggregated_results_name = 'results_rollout_gpu_{}_iters.csv'.format(n_iters)

    else:
        name = 'model{}_{}_iters.csv'
         aggregated_results_name = 'results_rollout_{}_iters.csv'.format(n_iters)
    
    os.chdir(directory)
    aggregated_results = []
    for i in range(1,7):
        model_name = name.format(i, n_iters)
        df = pd.read_csv(name)
        aggregated_data_this_model ={}
        aggregated_data_this_model['model'] = 'model{}'.format(i)
        aggregated_data_this_model['num_iters'] = n_iters
        aggregated_data_this_model['average_model_time_per_episode'] = df['total_model_time'].mean()
        aggregated_data_this_model['average_steps_per_episode'] = df['num_steps'].mean()
        aggregated_data_this_model['average_model_time_per_step'] = (df['total_model_time'].sum())/(df['num_steps'].sum())
        aggregated_data_this_model['average_reward_per_episode'] = df['reward'].mean()
        aggregated_data_this_model['min_reward'] = df['reward'].min()
        aggregated_data_this_model['max_reward'] = df['reward'].max()
        aggregated_results.append(aggregated_data_this_model)
    
    os.chdir(save_directory)
    with open(aggregated_results_name, mode='w+') as csv_agg_file:
        fieldnames = list(aggregated_results[0].keys())
        writer = csv.DictWriter(csv_agg_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results:
            writer.writerow(row) 
    
    return aggregated_results

def get_data(directory, num_iters):
    dir = dirname(dirname(abspath(__file__)))
    save_directory = '../result_analysis/rollout_results'
    aggregated_results_no_gpu = get_data_models(directory, num_iters, False, save_directory)
    aggregated_results_gpu = get_data_models(directory, num_iters, True, save_directory)

    model_names = [aggregated_results_no_gpu[i]['model'] for i in range(0,6)]
    vars = list(aggregated_results_gpu[0].keys())

    # plot data for non gpu 
    for var in vars:
        var_values_no_gpu=[aggregated_results_no_gpu[i][var] for i in range(0,6)]
        title_no_gpu = var + ' per model no gpu'
        save_name_no_gpu = dir + '/result_analysis/rollout_results/graphs/' + var + '_per_model_no_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_no_gpu, title_no_gpu, save_name_no_gpu)

        var_values_gpu = [aggregated_results_gpu[i][var] for i in range(0,6)]
        title_gpu = var + ' per model gpu'
        save_name_no_gpu = dir + '/result_analysis/rollout_results/graphs/' + var + '_per_model_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_gpu, title_gpu, save_name_gpu)

        title_combined = var + ' per model gpu and no gpu'
        save_name_combined = dir + '/result_analysis/rollout_results/graphs/' + var + '_per_model_no_gpu_and_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars_double(model_names, var_values_no_gpu, var_values_gpu, title_combined, save_name_combined, 'rollout without GPU', 'rollout with GPU')

    speedups = []
    for i in range(0,6):
        this_model_speedup = {}
        this_model_speedup['model'] = 'model{}'.format(i+1)
        for var in vars:
            this_model_speedup[var] = aggregated_results_no_gpu[i][var] / aggregated_results_gpu[i][var]
        speedups.append(this_model_speedup)
    
    ## plot speedup data
    for var in vars:
        var_values_speedup = [speedups[i][var] for i in range(0,6)]
        title = var + ' speedup no GPU vs GPU'
        save_name = dir + '/result_analysis/rollout_results/graphs/' + var + '_speedup_no_gpu_vs_gpu_it_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_speedup, title, save_name)

    with open(dir + '/result_analysis/rollout_results/results_speedup_no_gpu_vs_gpu_{}_iters.csv'.format(num_iters), mode='w+') as csv_speedup_file:
        fieldnames = list(speedups[0].keys())
        writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in speedups:
            writer.writerow(row)  

def main():
    directory = sys.argv[1]
    num_iters = int(sys.argv[2])
    get_data(directory, num_iters)         

if __name__ == "__main__":
    main()            
        
    

