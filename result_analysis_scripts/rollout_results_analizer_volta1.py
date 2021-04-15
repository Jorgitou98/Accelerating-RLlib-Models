
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

def get_data_models(name, aggregated_results_name, trained_model_descr, model_ids, directory, n_iters, save_directory):
    os.chdir(directory)
    aggregated_results = []
    cont = 0
    for i in model_ids:
        model_name = name.format(trained_model_descr[cont], i)
        df = pd.read_csv(model_name)
        aggregated_data_this_model ={}
        aggregated_data_this_model['model'] = i
        aggregated_data_this_model['num_iters'] = n_iters
        aggregated_data_this_model['average_model_time_per_episode'] = df['total_model_time'].mean()
        aggregated_data_this_model['average_steps_per_episode'] = df['num_steps'].mean()
        aggregated_data_this_model['average_model_time_per_step'] = (df['total_model_time'].sum())/(df['num_steps'].sum())
        aggregated_data_this_model['average_reward_per_episode'] = df['reward'].mean()
        aggregated_data_this_model['min_reward'] = df['reward'].min()
        aggregated_data_this_model['max_reward'] = df['reward'].max()
        aggregated_results.append(aggregated_data_this_model)
        cont+=1
    
    os.chdir(save_directory)
    with open(aggregated_results_name, mode='w+') as csv_agg_file:
        fieldnames = list(aggregated_results[0].keys())
        writer = csv.DictWriter(csv_agg_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results:
            writer.writerow(row) 
    
    return aggregated_results

def get_data(trained_model_descr, model_ids, directory, num_iters):
    descriptions = ['8_workers_gpu0', '8_workers_gpu1', '8_workers_both_gpus', '16_workers_both_gpus', '0_workers_gpu0', '0_workers_gpu1', '0_workers_both_gpus', '0_workers_gpu0_one_cpu', '0_workers_gpu1_one_cpu', '0_workers_both_gpus_one_cpu', '0_workers_no_gpus']
    dir = dirname(dirname(abspath(__file__)))
    save_directory = dir + '/result_analysis/rollout_results/volta1_{}_iters'.format(num_iters)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    names = ['trained_{}/model{}_' + i + '.csv' for i in descriptions]
    aggregated_names = ['results_volta1_{}_iters_{}.csv'.format(num_iters, i) for i in descriptions]
    aggregated_results = [get_data_models(names[i], aggregated_names[i], trained_model_descr, model_ids, directory, num_iters, save_directory) for i in range(0, len(descriptions))]

    model_names = ['model {}'.format(i) for i in model_ids]
    vars = ['average_model_time_per_episode', 'average_steps_per_episode', 'average_model_time_per_step', 'average_reward_per_episode', 'min_reward', 'max_reward']

    if not os.path.exists(dir + '/result_analysis/rollout_results/volta1_{}_iters/graphs/'.format(num_iters)):
        os.mkdir(dir + '/result_analysis/rollout_results/volta1_{}_iters/graphs/'.format(num_iters))

    for var in vars:
        var_values_list = []
        for desc in range(0, len(descriptions)):
            var_values=[aggregated_results[desc][i][var] for i in range(0, len(model_ids))]
            var_values_list.append(var_values)
            title = var + ' per model {} {} iters'.format(descriptions[desc], num_iters)
            save_name= dir + '/result_analysis/rollout_results/volta1_'+ str(num_iters) + '_iters/graphs/' + var + '_per_model_' + descriptions[desc] +'_' +str(num_iters) + '_iters.png'
            plot_results.plot_bars(model_names, var_values, title, save_name)


        title_combined = var + ' per model all configurations'
        save_name_combined = dir + '/result_analysis/rollout_results/volta1_' + str(num_iters) + '_iters/graphs/'+ var + '_per_model_all_descr.png' 
        plot_results.plot_bars_multiple(model_names, var_values_list, title_combined, save_name_combined, ['rollout {}'.format(desc) for desc in descriptions])

    for i in range(0, len(descriptions)-1):
        for j in range(i+1, len(descriptions)):
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
                save_name = dir + '/result_analysis/rollout_results/volta1_' + str(num_iters) + '_iters/graphs/'+ var + '_speedup_' + descriptions[i] + '_vs_'+ descriptions[j] + '_' + str(num_iters) + '_iters.png'
                plot_results.plot_bars(model_names, var_values_speedup, title, save_name)
 
            with open(dir + '/result_analysis/rollout_results/volta1_{}_iters/results_speedup_{}_vs_{}_{}_iters.csv'.format(num_iters,descriptions[i], descriptions[j], num_iters), mode='w+') as csv_speedup_file:
                fieldnames = list(speedups[0].keys())
                writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in speedups:
                    writer.writerow(row)  

def main():
    directory = sys.argv[1]
    num_iters = sys.argv[2]
    trained_model_descr_str = sys.argv[3]
    trained_model_descr= trained_model_descr_str[1:len(trained_model_descr_str)-1].split(',')
    model_ids_str = sys.argv[4]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    get_data(trained_model_descr, model_ids, directory, num_iters)         

if __name__ == "__main__":
    main()            
        
    

