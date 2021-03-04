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

def merge_csv(files_to_merge, merged_name):
    combined_csv = pd.concat([pd.read_csv(f) for f in files_to_merge])
    combined_csv.sort_values(by='training_iteration', inplace = True)
    combined_csv.drop_duplicates(keep = 'first', inplace = True)
    combined_csv.to_csv(merged_name, index=False, encoding='utf-8-sig') 


def get_data_models(directory,it_ini, it_fin, policy, model_name_format, model_name_all_format, name_split_len, aggregated_results_name, save_directory):
    aggregated_results = []
    os.chdir(directory)
    for i in range(1,7):
        model_name = model_name_format.format(i)
        model_name_all = model_name_all_format.format(i)     
        if(len([j for j in glob.glob(model_name)]) != 1):
            model_names_list = [k for k in glob.glob(model_name_all)]
            final_model_name_list = []
            for name in model_names_list:
                splitted_name = name.split('_')
                if(len(splitted_name) == name_split_len and int(splitted_name[name_split_len-2]) >= it_ini and int(splitted_name[name_split_len-2]) <= it_fin and int(splitted_name[name_split_len-1]) <= it_fin and int(splitted_name[name_split_len-1]) >= it_ini):
                    final_model_name_list.append(name + '/progress.csv')
            if not os.path.exists(model_name):
                os.mkdir(model_name)
            merge_csv(final_model_name_list, directory + '/' + model_name + '/progress.csv')
        df = pd.read_csv(model_name + '/progress.csv')
        df.dropna(inplace = True)
        aggregated_data_this_model = {}
        aggregated_data_this_model['model'] = 'model' + str(i)
        aggregated_data_this_model['iter_ini'] = it_ini
        aggregated_data_this_model['iter_fin'] = it_fin
        aggregated_data_this_model['reward_mean_last_iter'] = df[df['training_iteration'] == df['training_iteration'].max()]['episode_reward_mean'].values[0]
        aggregated_data_this_model['len_mean_last_iter'] = df[df['training_iteration'] == df['training_iteration'].max()]['episode_len_mean'].values[0]
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
   

        
def get_data(directory, it_ini, it_fin, it_ini_gpu, it_fin_gpu, policy):
    dir = dirname(dirname(abspath(__file__)))
    print(dir)
    name_gpu = 'model{}_' + policy+ '_gpu_it_' + str(it_ini) + '_' + str(it_fin)
    name_no_gpu = 'model{}_' + policy + '_it_' + str(it_ini) + '_' + str(it_fin)
    save_directory = '../result_analysis/training_results'
    aggregated_results_name_gpu = 'results_{}_gpu_it_{}_{}.csv'.format(policy, it_ini_gpu, it_fin_gpu)
    aggregated_results_name_no_gpu = 'results_{}_it_{}_{}.csv'.format(policy, it_ini, it_fin)
    model_name_all_gpu = 'model{}_' + policy+ '_gpu_it_*'
    model_name_all_no_gpu = 'model{}_' + policy+ '_it_*'
    name_split_len_gpu = 6
    name_split_len_no_gpu = 5
    aggregated_results_no_gpu = get_data_models(directory,it_ini, it_fin, policy, name_no_gpu, model_name_all_no_gpu, name_split_len_no_gpu, aggregated_results_name_no_gpu, save_directory)
    aggregated_results_gpu = get_data_models(directory,it_ini_gpu, it_fin_gpu, policy, name_gpu, model_name_all_gpu, name_split_len_gpu, aggregated_results_name_gpu, save_directory)
    
    model_names = [aggregated_results_no_gpu[i]['model'] for i in range(0,6)]
    vars = ['mean_sample_time_ms','mean_sample_throughput','mean_load_time_ms','mean_load_throughput','mean_learn_time_ms','mean_learn_throughput','mean_update_time_ms', 'mean_ram_util_percent', 'mean_cpu_util_percent']
    
    ## plot data for non gpu training
    for var in vars:
        var_values_no_gpu = [aggregated_results_no_gpu[i][var] for i in range(0,6)]
        title_no_gpu = var + ' per model no gpu'
        save_name_no_gpu = dir + '/result_analysis/training_results/graphs/' + var + '_per_model_no_gpu_it_' + str(it_ini) + '_'+ str(it_fin) + '.png'
        plot_results.plot_bars(model_names, var_values_no_gpu, title_no_gpu, save_name_no_gpu)

        var_values_gpu = [aggregated_results_gpu[i][var] for i in range(0,6)]
        title_gpu = var + ' per model gpu'
        save_name_gpu = dir + '/result_analysis/training_results/graphs/' + var + '_per_model_gpu_it_' + str(it_ini_gpu) + '_'+ str(it_fin_gpu) + '.png'
        plot_results.plot_bars(model_names, var_values_gpu, title_gpu, save_name_gpu)

        title_combined = var + ' per model gpu and no gpu'
        save_name_combined = dir + '/result_analysis/training_results/graphs/' + var+ '_per_model_no_gpu_it_' + str(it_ini) + '_'+ str(it_fin) + '_and_gpu_it_' + str(it_ini_gpu) + '_' + str(it_fin_gpu) + '.png'
        plot_results.plot_bars_double(model_names, var_values_no_gpu, var_values_gpu, title_combined, save_name_combined, 'training without GPU', 'training with GPU')
    
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
        save_name = dir + '/result_analysis/training_results/graphs/' + var + '_speedup_no_gpu_it_' + str(it_ini) + '_' + str(it_fin) + '_vs_gpu_it_' + str(it_ini_gpu) + '_' + str(it_fin_gpu) + '.png'
        plot_results.plot_bars(model_names, var_values_speedup, title, save_name)

    with open(dir + '/result_analysis/training_results/results_speedup_{}_no_gpu_it_{}_{}_vs_gpu_it_{}_{}.csv'.format(policy, it_ini, it_fin, it_ini_gpu, it_fin_gpu), mode='w+') as csv_speedup_file:
        fieldnames = list(speedups[0].keys())
        writer = csv.DictWriter(csv_speedup_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in speedups:
            writer.writerow(row)  

def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    it_ini_gpu = int(sys.argv[4])
    it_fin_gpu = int(sys.argv[5])
    policy = sys.argv[6]
    get_data(directory, it_ini, it_fin, it_ini_gpu, it_fin_gpu, policy)         

if __name__ == "__main__":
    main()        
