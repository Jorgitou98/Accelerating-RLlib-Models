import training_results_analizer
import os
import sys
import pandas as pd
import plot_results


def combine_data(directory,it_ini, it_fin, model_ids):
    os.chdir(directory)
    name = "model{}_ppo_gpu_it_{}_{}/progress.csv"
    for model in model_ids:
        file_names = []
        for i in range(it_ini, it_fin, 1000):
            file_names.append(name.format(model, i+1, i+1000))
        combined_name = "../ray_results_gathered/{}_iters/model{}_ppo_gpu_it_{}_{}.csv".format(it_fin, model, it_ini+1, it_fin)
        training_results_analizer.merge_csv(file_names, combined_name)

def plot_data(it_ini, it_fin, model_ids):
    os.chdir('ray_results_gathered/{}_iters'.format(it_fin))
    file_names = ['model{}_ppo_gpu_it_{}_{}.csv'.format(i,it_ini+1,it_fin) for i in[1,3,4]]
    df_list = []
    for i in range(0,len(model_ids)):
        df_list.append(pd.read_csv(file_names[i]))

    #Compare the three models results
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
    x_values = [i for i in range(it_ini+1, it_fin+1)]
    for var in vars_to_compare:
        y_values = []
        for i in range(0,len(model_ids)):
            y_values.append(list(df_list[i][var]))
        labels = ['model{}'.format(i) for i in model_ids]
        models_str_title=' models'
        models_str_save ='_models'
        for i in model_ids:
            models_str_title = models_str_title + ' ' + i
            models_str_save = models_str_save + '_' + i

        title = 'Compare ' + var + models_str_title
        var_save_name = var.split('/')[len(var.split('/'))-1]
        save_name = 'graphs/compare_' + var_save_name + '_it_' + str(it_ini + 1) + '_' + str(it_fin) + models_str_save + '.png'
        plot_results.plot_line_three(x_values, y_values, labels, title, save_name)

    # See rewards evolution for each model
    for i in range(0,len(model_ids)):
        y_values = []
        for var in ['episode_reward_max','episode_reward_min','episode_reward_mean']:
            y_values.append(list(df_list[i][var]))
        labels = ['max reward','min reward','average reward']
        title = 'Rewards Model {} iterations {} to {}'.format(model_ids[i], it_ini+1, it_fin)
        save_name = 'graphs/rewards_model_{}_it_{}_{}.png'.format(model_ids[i], it_ini+1, it_fin)
        plot_results.plot_line_three(x_values, y_values, labels, title, save_name)




def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    model_ids_str = sys.argv[4]
    model_ids = model_ids_str[1:len(model_ids_str)-1].split(',')
    combine_data(directory,it_ini,it_fin, model_ids)
    plot_data(it_ini, it_fin, model_ids)
    

if __name__ == '__main__':
    main()


