import os
import pickle

data_x3dmodel_1 = 'training_metrics.pkl'

with open(data_x3dmodel_1, 'rb')as file:
    loaded_data = pickle.load(file)
    print(loaded_data)