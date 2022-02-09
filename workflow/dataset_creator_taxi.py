
import sys
import numpy as np
dataset_name = sys.argv[1]

with open(dataset_name + '.npy', 'wb') as f:
    for one_hot in np.eye(500):
        print(one_hot)
        np.save(f, one_hot)






