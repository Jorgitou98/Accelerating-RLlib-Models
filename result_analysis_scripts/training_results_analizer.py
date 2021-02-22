import csv
import glob
import os
import pandas as pd

def merge_csv(files_to_merge, merged_name):
    combined_csv = pd.concat([pd.read_csv(f) for f in files_to_merge])
    combined_csv.to_csv( merged_name, index=False, encoding='utf-8-sig')

def get_data(directory,it_ini, it_fin, policy):
    os.chdir(directory)
    name = 'model{}_{}_gpu_it_' + str(it_ini) + '_' + str(it_fin)
    for i in range(1,7):
        model_name = name.format(i, policy)
        
        if(len([j for j in glob.glob(name)]) != 1):
            model_names_list = [k for k in glob.glob('model_{}_gpu_it_*'.format(i))]
            final_model_name_list = []
            for name in model_names_list:
                splitted_name = name.split('_')
                if(len(splitted_name) == 6 and splitted_name[4] >= it_ini and splitted_name[4] <= it_fin and splitted_name[5] <= it_fin and splitted_name[5] >= it_ini):
                    final_model_name_list.append(name + '/progress.csv')
            merge_csv(final_model_name_list, directoy + '/' + name + '/progress.csv')

def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    policy = sys.argv[4]
    get_data(directory, it_ini, it_fin, policy)         
        
