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



def get_data_models(directory, model_names, model_names_short, aggregated_results_name, save_directory):
    aggregated_results = []
    os.chdir(directory)
    for i in range (0, len(model_names)):        
        df = pd.read_csv(model_names[i] + '/progress.csv')
        df.dropna(inplace = True)
        aggregated_data_this_model = {}
        aggregated_data_this_model['model'] = model_names_short[i]
        aggregated_data_this_model['reward_mean_last_iter'] = df[df['training_iteration'] == df['training_iteration'].max()]['episode_reward_mean'].values[0]
        aggregated_data_this_model['len_mean_last_iter'] = df[df['training_iteration'] == df['training_iteration'].max()]['episode_len_mean'].values[0]
        aggregated_data_this_model['mean_time_this_iter_s'] = df['time_this_iter_s'].mean()
        aggregated_data_this_model['mean_learn_time_ms'] = df['timers/learn_time_ms'].mean()
        aggregated_data_this_model['mean_learn_throughput'] = df['timers/learn_throughput'].mean()
        aggregated_data_this_model['mean_sample_time_ms'] = df['timers/sample_time_ms'].mean()
        aggregated_data_this_model['mean_sample_throughput'] = df['timers/sample_throughput'].mean()
        aggregated_data_this_model['mean_load_time_ms'] = df['timers/load_time_ms'].mean()
        aggregated_data_this_model['mean_load_throughput'] = df['timers/load_throughput'].mean()
        aggregated_data_this_model['mean_update_time_ms'] = df['timers/update_time_ms'].mean()
        aggregated_data_this_model['mean_cpu_util_percent'] = df['perf/cpu_util_percent'].mean()
        aggregated_data_this_model['mean_ram_util_percent'] = df['perf/ram_util_percent'].mean()
        
        aggregated_results.append(aggregated_data_this_model)

    
    

    os.chdir(save_directory)
    with open(aggregated_results_name, mode='w+') as csv_agg_file:
        fieldnames = list(aggregated_results[0].keys())
        writer = csv.DictWriter(csv_agg_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results:
            writer.writerow(row)  


    return aggregated_results
   

        
def get_data(directory, model_names, model_names_short, model, it_ini, it_fin):
    dir = dirname(dirname(abspath(__file__)))
    print(dir)
    save_directory = '../../result_analysis/training_results/volta1_def'
    aggregated_results_name = 'results_model_{}_it_{}_{}.csv'.format(model, it_ini, it_fin)
    aggregated_results = get_data_models(directory, model_names, model_names_short, aggregated_results_name, save_directory)
    
    vars = ['mean_time_this_iter_s','mean_sample_time_ms','mean_sample_throughput','mean_load_time_ms','mean_load_throughput','mean_learn_time_ms','mean_learn_throughput','mean_update_time_ms', 'mean_ram_util_percent', 'mean_cpu_util_percent']
    y_labels['time(s)', 'time(ms)', None, 'time(ms)', None, 'time(ms)', None, 'time(ms)', '% util', '% util']
    
    for i in range(len(vars)):
        var_values = [aggregated_results[i][vars[i]] for i in range(0,len(model_names))]
        title= vars[i] + ' model {}'.format(model)
        save_name = dir + '/result_analysis/training_results/volta1_def/graphs/' + vars[i] + '_model{}_it_'.format(model) + str(it_ini) + '_'+ str(it_fin) + '.png'
        plot_results.plot_bars(model_names_short, var_values, title, save_name, y_label=y_labels[i])

def plot_data(directory, model_names, model_names_short, model, it_ini, it_fin):
    os.chdir(directory)
    df_list = []
    for i in range(0,len(model_names)):
        df_list.append(pd.read_csv(model_names[i] + '/progress.csv'))

    os.chdir('../../result_analysis/training_results/volta1_def')
    #Compare all models results
    vars_to_compare = (['episode_reward_max','episode_reward_min','episode_reward_mean',
    'episode_len_mean','episodes_this_iter','timesteps_total','done','episodes_total',
    'training_iteration','time_this_iter_s','time_total_s','time_since_restore',
    'timesteps_since_restore','iterations_since_restore','timers/sample_time_ms',
    'timers/sample_throughput','timers/load_time_ms','timers/load_throughput','timers/learn_time_ms',
    'timers/learn_throughput','timers/update_time_ms',
    'info/num_steps_sampled','info/num_steps_trained','perf/cpu_util_percent','perf/ram_util_percent',
    'info/learner/default_policy/cur_kl_coeff','info/learner/default_policy/cur_lr',
    'info/learner/default_policy/total_loss','info/learner/default_policy/policy_loss',
    'info/learner/default_policy/vf_loss','info/learner/default_policy/vf_explained_var',
    'info/learner/default_policy/kl','info/learner/default_policy/entropy',
    'info/learner/default_policy/entropy_coeff'])
    x_values = [i for i in range(it_ini, it_fin+1)]
    for var in vars_to_compare:
        y_values = []
        for i in range(0,len(model_names)):
            y_values.append(list(df_list[i][var]))

        title = 'Compare ' + var + ' model {}'.format(model)
        var_save_name = var.split('/')[len(var.split('/'))-1]
        save_name = 'graphs/compare_' + var_save_name + '_it_' + str(it_ini) + '_' + str(it_fin) + '_model{}'.format(model) + '.png'
        plot_results.plot_line_multiple(x_values, y_values, model_names_short, title, save_name, len(model_names))

    # See rewards evolution for each model
    for i in range(0,len(model_names)):
        y_values = []
        for var in ['episode_reward_max','episode_reward_min','episode_reward_mean']:
            y_values.append(list(df_list[i][var]))
        labels = ['max reward','min reward','average reward']
        title = 'Rewards Model {} {} iterations {} to {}'.format(model, model_names_short[i], it_ini, it_fin)
        save_name = 'graphs/rewards_{}.png'.format(model_names[i])
        plot_results.plot_line_multiple(x_values, y_values, labels, title, save_name, 3)
    


def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    model = int(sys.argv[4])
    model_names_str = sys.argv[5]
    model_names = model_names_str[1:len(model_names_str)-1].split(',')
    model_names_short_str = sys.argv[6]
    model_names_short = model_names_short_str[1:len(model_names_short_str)-1].split(',')
    get_data(directory, model_names, model_names_short, model, it_ini, it_fin)
    plot_data(directory, model_names, model_names_short, model, it_ini, it_fin)         

if __name__ == "__main__":
    main()        
