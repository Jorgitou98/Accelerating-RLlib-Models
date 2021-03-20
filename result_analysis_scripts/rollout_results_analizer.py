
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

def get_data(desc1, desc2, model_ids, directory, num_iters):
    dir = dirname(dirname(abspath(__file__)))
    save_directory = dir + '/result_analysis/rollout_results'
    name1 = 'model{}_' + desc1 + '.csv'
    name2 = 'model{}_' + desc2 + '.csv'
    aggregated_name1 = 'results_{}_iters_{}.csv'.format(num_iters, desc1)
    aggregated_name2 = 'results_{}_iters_{}.csv'.format(num_iters, desc2)
    aggregated_results1 = get_data_models(name2, aggregated_name2, model_ids, directory, num_iters, save_directory)
    aggregated_results2 = get_data_models(name2, aggregated_name2, model_ids, directory, num_iters, save_directory)

    model_names = [aggregated_results1[i]['model'] for i in range(0, len(model_ids))]
    vars = ['average_model_time_per_episode', 'average_steps_per_episode', 'average_model_time_per_step', 'average_reward_per_episode', 'min_reward', 'max_reward']


    for var in vars:
        var_values1=[aggregated_results1[i][var] for i in range(0, len(model_ids))]
        title1 = var + ' per model {} {} iters'.format(desc1, num_iters)
        save_name1= dir + '/result_analysis/rollout_results/graphs/'+ str(num_iters) + '_iters/' + var + '_per_model_' + desc1 +'_' +str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values1, title1, save_name1)

        var_values2 = [aggregated_results2[i][var] for i in range(0, len(model_ids))]
        title2 = var + ' per model {} {} iters'.format(desc2, num_iters)
        save_name2 = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/' + var + '_per_model_' + desc2 + '_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values2, title2, save_name2)

        title_combined = var + ' per model {} and {} '.format(desc1, desc2)
        save_name_combined = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/'+ var + '_per_model_' + desc1 + '_and_' + desc2 + '_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars_double(model_names, var_values1, var_values2, title_combined, save_name_combined, 'rollout {}'.format(desc1), 'rollout {}'.format(desc2))

    speedups = []
    for i in range(0, len(model_ids)):
        this_model_speedup = {}
        this_model_speedup['model'] = 'model{}'.format(model_ids[i])
        for var in vars:
            this_model_speedup[var] = aggregated_results1[i][var] / aggregated_results2[i][var]
        speedups.append(this_model_speedup)
    
    ## plot speedup data
    for var in vars:
        var_values_speedup = [speedups[i][var] for i in range(0,len(model_ids))]
        title = var + ' speedup {} vs {}'.format(desc1, desc2)
        save_name = dir + '/result_analysis/rollout_results/graphs/' + str(num_iters) + '_iters/'+ var + '_speedup_' + desc1 + '_vs_'+ desc2 + '_' + str(num_iters) + '_iters.png'
        plot_results.plot_bars(model_names, var_values_speedup, title, save_name)

    with open(dir + '/result_analysis/rollout_results/results_speedup_{}_vs_{}_{}_iters.csv'.format(desc1, desc2, num_iters), mode='w+') as csv_speedup_file:
        fieldnames = list(speedups[0].keys())
        writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in speedups:
            writer.writerow(row)  

def main():
    directory = sys.argv[1]
    desc1 = sys.argv[2]
    desc2 = sys.argv[3]
    num_iters = int(sys.argv[4])
    model_ids_str = sys.argv[5]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    get_data(desc1, desc2, model_ids, directory, num_iters)         

if __name__ == "__main__":
    main()            
        
    

