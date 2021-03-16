import training_results_analizer

def combine_data(directory,it_ini, it_fin):
    file_names = []
    os.chdir(directory)
    name = "model{}_ppo_gpu_it_{}_{}/progress.csv"
    for model in [1,3,4]:
        for i in range(it_ini, it_fin+1, 1000):
            file_names.append(name.format(model, i+1, i+1000))
        combined_name = "../ray_results_gathered/model{}_ppo_gpu_it_{}_{}.csv".format(model, it_ini+1, it_fin)
        training_results_analizer.merge_csv(file_names, combined_name)




def main():
    directory = sys.argv[1]
    it_ini = int(sys.argv[2])
    it_fin = int(sys.argv[3])
    combine_data(directory,it_ini,it_fin)

if __name == '__main__':
    main()


