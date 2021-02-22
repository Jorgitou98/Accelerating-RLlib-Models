import csv
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def merge_csv(files_to_merge, merged_name):
    combined_csv = pd.concat([pd.read_csv(f) for f in files_to_merge])
    combined_csv.sort_values(by='training_iteration', inplace = True)
    combined_csv.drop_duplicates(keep = 'first', inplace = True)
    combined_csv.to_csv(merged_name, index=False, encoding='utf-8-sig')

def plot_line(df, columnX, columnY, name, save_name, x_label = None, y_label = None):
    plt.close('all')
    df.show()
    x_values = [row[0] for row in df.select(columnX).collect()]
    y_values = [row[0] for row in df.select(columnY).collect()]
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_title(name, loc='center', wrap=True)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name)
    print('Graph saved at ' + save_name)

def get_data_models(directory,it_ini, it_fin, policy, model_name, model_name_all, name_split_len, aggregated_results_name):
    aggregated_results = []
    os.chdir(directory)
    for i in range(1,7):
        model_name = model_name.format(i)
        model_name_all = model_name_all.format(i)     
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
        #vars_to_plot = ['episode_reward_mean', 'episode_reward_max', 'episode_reward_mean']
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
        aggregated_results.append(aggregated_data_this_model)

    with open(aggregated_results_name, mode='w') as csv_agg_file:
        fieldnames = list(aggregated_results[0].keys())
        writer = csv.DictWriter(csv_agg_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results:
            writer.writerow(row)  

def get_data(directory, it_ini, it_fin, it_ini_gpu, it_fin_gpu, policy):
    name_gpu = ('model{}_' + policy+ '_gpu_it_' + str(it_ini) + '_' + str(it_fin))
    name_no_gpu = name_gpu = ('model{}_' + policy+ '_it_' + str(it_ini) + '_' + str(it_fin))
    aggregated_results_name_gpu = '~/Mejorando-el-Aprendizaje-Automatico/result_analysis/training_results/results_{}_gpu_it_{}_{}.csv'.format(policy, it_ini_gpu, it_fin_gpu)
    aggregated_results_name_no_gpu = '~/Mejorando-el-Aprendizaje-Automatico/result_analysis/training_results/results_{}_it_{}_{}.csv'.format(policy, it_ini, it_fin)
    model_name_all_gpu = 'model{}_' + policy+ '_gpu_it_*'
    model_name_all_no_gpu = 'model{}_' + policy+ '_it_*'
    name_split_len_gpu = 6
    name_split_len_no_gpu = 5
    get_data_models(directory,it_ini, it_fin, policy,name_no_gpu, model_name_all_no_gpu, name_split_len_no_gpu, aggregated_results_name_no_gpu)
    get_data_models(directory,it_ini_gpu, it_fin_gpu, policy,name_gpu, model_name_all_gpu, name_split_len_gpu, aggregated_results_name_gpu)



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
