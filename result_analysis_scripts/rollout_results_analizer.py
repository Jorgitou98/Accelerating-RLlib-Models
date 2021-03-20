
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

def get_data(descriptions, model_ids, directory, num_iters):
    dir = dirname(dirname(abspath(__file__)))
    save_directory = dir + '/result_analysis/rollout_results/{}_iters'.format(num_iters)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    names = ['model{}_' + i + '.csv' for i in descriptions]
    aggregated_names = ['results_{}_iters_{}.csv'.format(num_iters, i) for i in descriptions]
    aggregated_results = [get_data_models(names[i], aggregated_names[i], model_ids, directory, num_iters, save_directory) for i in range(0, len(descriptions))]

    model_names = ['model{}'.format(i) for i in model_ids]
    vars = ['average_model_time_per_episode', 'average_steps_per_episode', 'average_model_time_per_step', 'average_reward_per_episode', 'min_reward', 'max_reward']

    if not os.path.exists(dir + '/result_analysis/rollout_results/' + str(num_iters) + '_iters/graphs/'):
        os.mkdir(dir + '/result_analysis/rollout_results/' + str(num_iters) + '_iters/graphs/')

    for var in vars:
        var_values_list = []
        for desc in range(0, len(descriptions)):
            var_values=[aggregated_results[desc][i][var] for i in range(0, len(model_ids))]
            var_values_list.append(var_values)
            title = var + ' per model {} {} iters'.format(descriptions[desc], num_iters)
            save_name= dir + '/result_analysis/rollout_results/'+ str(num_iters) + '_iters/graphs/' + var + '_per_model_' + descriptions[desc] +'_' +str(num_iters) + '_iters.png'
            plot_results.plot_bars(model_names, var_values, title, save_name)


        title_combined = var + ' per model'
        save_name_combined = dir + '/result_analysis/rollout_results/' + str(num_iters) + '_iters/graphs/'+ var + '_per_model_' 
        for desc in range(0, len(descriptions)):
            if(desc == len(descriptions)-1):
                title_combined += (' and ' + descriptions[desc])
                save_name_combined +=('_and_' + descriptions[desc])
            else:
                title_combined += (', ' + descriptions[desc])
                save_name_combined +=('_' + descriptions[desc])
        save_name_combined += (str(num_iters) + '_iters.png')
        plot_results.plot_bars_multiple(model_names, var_values_list, title_combined, save_name_combined, ['rollout {}'.format(desc) for desc in descriptions])

    for i in range(0, len(descriptions)):
        for j in range(i, len(descriptions)):
            speedups = []
            for k in range(0, len(model_ids)):
                this_model_speedup = {}
                this_model_speedup['model'] = 'model{}'.format(model_ids[k])
                for var in vars:
                    this_model_speedup[var] = aggregated_results[i][k][var] / aggregated_results[j][k][var]
                speedups.append(this_model_speedup)
            ## plot speedup data
            for var in vars:
                var_values_speedup = [speedups[m][var] for m in range(0,len(model_ids))]
                title = var + ' speedup {} vs {}'.format(descriptions[i], descriptions[j])
                save_name = dir + '/result_analysis/rollout_results/' + str(num_iters) + '_iters/graphs/'+ var + '_speedup_' + descriptions[i] + '_vs_'+ descriptions[j] + '_' + str(num_iters) + '_iters.png'
                plot_results.plot_bars(model_names, var_values_speedup, title, save_name)
 
            with open(dir + '/result_analysis/rollout_results/{}_iters/results_speedup_{}_vs_{}_{}_iters.csv'.format(num_iters,descriptions[i], descriptions[j], num_iters), mode='w+') as csv_speedup_file:
                fieldnames = list(speedups[0].keys())
                writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in speedups:
                    writer.writerow(row)  

def main():
    directory = sys.argv[1]
    descriptions_str = sys.argv[2]
    descriptions = descriptions_str[1:len(descriptions_str)-1].split(',')
    num_iters = int(sys.argv[3])
    model_ids_str = sys.argv[4]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    get_data(descriptions, model_ids, directory, num_iters)         

if __name__ == "__main__":
    main()            
        
    

