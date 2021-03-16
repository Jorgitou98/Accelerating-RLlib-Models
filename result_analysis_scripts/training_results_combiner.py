import training_results_analizer
import os
import sys
import pandas as pd
import plot_results


def combine_data(directory,it_ini, it_fin):
    os.chdir(directory)
    name = "model{}_ppo_gpu_it_{}_{}/progress.csv"
    for model in [1,3,4]:
        file_names = []
        for i in range(it_ini, it_fin, 1000):
            file_names.append(name.format(model, i+1, i+1000))
        combined_name = "../ray_results_gathered/model{}_ppo_gpu_it_{}_{}.csv".format(model, it_ini+1, it_fin)
        training_results_analizer.merge_csv(file_names, combined_name)

def plot_data(it_ini, it_fin):
    os.chdir('ray_results_gathered')
    file_names = ['model{}_ppo_gpu_it_{}_{}.csv'.format(i,it_ini+1,it_fin) for i in[1,3,4]]
    df_list = []
    for i in range(0,3):
        df_list.append(pd.read_csv(file_names[i]))

    #Compare the three models results
    vars_to_compare = list(df_list[0].columns)
    x_values = [i for i in range(it_ini+1, it_fin+1)]
    for var in vars_to_compare:
        y_values = []
        for i in range(0,3):
            y_values.append(list(df_list[i][var]))
        labels = ['model1', 'model3', 'model4']
        title = 'Compare ' + var + ' models 1,3 and 4'
        var_save_name = var.split('/')[len(var.split('/'))-1]
        save_name = 'graphs/compare_' + var_save_name + '_it_' + str(it_ini + 1) + '_' + str(it_fin) +'_models_1_3_4.png'
        plot_results.plot_line_three(x_values, y_values, labels, title, save_name)

    # See rewards evolution for each model
    models=[1,3,4]
    for i in range(0,3):
        y_values = []
        for var in ['episode_reward_max','episode_reward_min','episode_reward_mean']:
            y_values.append(list(df_list[i][var]))
        labels = ['max reward','min reward','average reward']
        title = 'Rewards Model {} iterations {} to {}'.format(models[i], it_ini+1, it_fin)
        save_name = 'graphs/rewards_model_{}_it_{}_{}.png'.format(models[i], it_ini+1, it_fin)
        plot_results.plot_line_three(x_values, y_values, labels, totle, save_name)




def main():
    '''
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    combine_data(directory,it_ini,it_fin)
    '''
    it_ini = int(sys.argv[1])
    it_fin = int(sys.argv[2])
    plot_data(it_ini, it_fin)
    

if __name__ == '__main__':
    main()


