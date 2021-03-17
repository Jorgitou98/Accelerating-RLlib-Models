
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

def get_data_models(name, aggregated_results_name, model_ids, directory, n_iters, save_directory):
    os.chdir(directory)
    aggregated_results = []
    for i in model_ids:
        model_name = name.format(i, n_iters)
        df = pd.read_csv(model_name)
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

def get_data(num_workers_no_gpu, num_workers_gpu, model_ids, directory, num_iters):
    dir = dirname(dirname(abspath(__file__)))
    save_directory = dir + '/result_analysis/rollout_results'
    name_gpu = 'model{}_' + str(num_workers_gpu) + '_workers_gpu.csv'
    name_no_gpu = 'model{}_' + str(num_workers_no_gpu) + '_workers_no_gpu.csv'
    aggregated_name_gpu = 'results_{}_iters_{}_workers_gpu.csv'.format(num_iters, num_workers_gpu)
    aggregated_name_no_gpu = 'results_{}_iters_{}_workers_no_gpu.csv'.format(num_iters, num_workers_no_gpu)
    aggregated_results_no_gpu = get_data_models(name_no_gpu, aggregated_name_no_gpu, model_ids, directory, num_iters, save_directory)
    aggregated_results_gpu = get_data_models(name_gpu, aggregated_name_gpu, model_ids, directory, num_iters, save_directory)

    model_names = [aggregated_results_no_gpu[i]['model'] for i in range(0, len(model_ids))]
    vars = ['average_model_time_per_episode', 'average_steps_per_episode', 'average_model_time_per_step', 'average_reward_per_episode', 'min_reward', 'max_reward']

    # plot data for non gpu 
    for var in vars:
        var_values_no_gpu=[aggregated_results_no_gpu[i][var] for i in range(0, len(model_ids))]
        title_no_gpu = var + ' per model {} workers no gpu {} iters'.format(num_workers_no_gpu, num_iters)
        save_name_no_gpu = dir + '/result_analysis/rollout_results/graphs/'+ str(num_iters) + '_iters/' + var + '_per_model_' + str(num_workers_no_gpu) + '_workers_no_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_no_gpu, title_no_gpu, save_name_no_gpu)

        var_values_gpu = [aggregated_results_gpu[i][var] for i in range(0, len(model_ids))]
        title_gpu = var + ' per model gpu {} workers gpu {} iters'.format(num_workers_gpu, num_iters)
        save_name_gpu = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/' + var + '_per_model_' +str(num_workers_gpu) + '_workers_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_gpu, title_gpu, save_name_gpu)

        title_combined = var + ' per model {} workers gpu and {} workers no gpu'.format(num_workers_gpu, num_workers_no_gpu)
        save_name_combined = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/'+ var + '_per_model_' + str(num_workers_no_gpu) + '_workers_no_gpu_and_' + str(num_workers_gpu) + '_workers_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars_double(model_names, var_values_no_gpu, var_values_gpu, title_combined, save_name_combined, 'rollout {} workers without GPU'.format(num_workers_no_gpu), 'rollout {} workers with GPU'.format(num_workers_gpu))

    speedups = []
    for i in range(0, len(model_ids)):
        this_model_speedup = {}
        this_model_speedup['model'] = 'model{}'.format(model_ids[i])
        for var in vars:
            this_model_speedup[var] = aggregated_results_no_gpu[i][var] / aggregated_results_gpu[i][var]
        speedups.append(this_model_speedup)
    
    ## plot speedup data
    for var in vars:
        var_values_speedup = [speedups[i][var] for i in range(0,len(model_ids))]
        title = var + ' speedup {} workers no GPU vs {} workers GPU'.format(num_workers_no_gpu, num_workers_gpu)
        save_name = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/'+ var + '_speedup_' + str(num_workers_no_gpu) + '_workers_no_gpu_vs_'+ str(num_workers_gpu) + '_workers_gpu_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_speedup, title, save_name)

    with open(dir + '/result_analysis/rollout_results/results_speedup_{}_workers_no_gpu_vs_{}_workers_gpu_{}_iters.csv'.format(num_workers_no_gpu, num_workers_gpu, num_iters), mode='w+') as csv_speedup_file:
        fieldnames = list(speedups[0].keys())
        writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in speedups:
            writer.writerow(row)  

def main():
    directory = sys.argv[1]
    num_workers_no_gpu = int(sys.argv[2])
    num_workers_gpu = int(sys.argv[3])
    num_iters = int(sys.argv[4])
    model_ids_str = sys.argv[5]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    get_data(num_workers_no_gpu, num_workers_gpu, model_ids, directory, num_iters)         

if __name__ == "__main__":
    main()            
        
    

