import numpy as np
import matplotlib.pyplot as plt
import csv, os

path = "tmp/"

def ini_csv(csv_path=''):
    global path
    path = csv_path + "tmp/"
    if not os.path.exists(path):
      os.mkdir(path)
    header = ['timesteps', 'reward']
    data = open(path+"training_data.csv", "a", encoding='UTF8', newline='')
    
    writer = csv.writer(data)
    
    if os.stat(path+"training_data.csv").st_size == 0: 
        writer.writerow(header)
    
# designed to receive only new data - to improve efficiency.
def save_in_csv(x,y):
    z = np.append(x.reshape(x.size, 1), y.reshape(y.size, 1), axis=1)

    data = open(path+"training_data.csv", "a", encoding='UTF8', newline='')    
    writer = csv.writer(data)
    writer.writerows(z)