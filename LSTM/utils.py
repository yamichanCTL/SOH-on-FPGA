import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# network structure
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x) # shape（seq_Len, batch, num_directions * hidden_size)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出作为全连接层的输入
        # print(out.shape)
        return out

def load_data(battery):
    mat = loadmat('../datasets/BatteryDataset/' + battery + '.mat') # MAT PATH ESSENTIAL
    counter = 0
    dataset = []
    capacity_data = []
    # print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0])) #total cycle

    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                          int(row['time'][0][1]),
                                          int(row['time'][0][2]),
                                          int(row['time'][0][3]),
                                          int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            for j in range(len(data[0][0]['Voltage_measured'][0])):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,temperature_measured,
                                current_load, voltage_load, time])

            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity,
                                  voltage_measured, current_measured,temperature_measured,
                                  current_load, voltage_load, time])
            counter = counter + 1
    # print(dataset[0])

    return [pd.DataFrame(data=dataset,
                         columns=['cycle', 'ambient_temperature', 'datetime','capacity',
                                  'voltage_measured','current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                         columns=['cycle', 'ambient_temperature', 'datetime','capacity',
                                  'voltage_measured','current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time'])]

def norm_data(battery_id):
    dataset, capacity = load_data(battery_id)
    C = capacity['capacity'][0]
    soh = []
    for i in range(len(capacity)):
        soh.append([capacity['capacity'][i] / C])
    soh = pd.DataFrame(data=soh, columns=['SOH'])

    # features for training
    attribs=['voltage_measured', 'current_measured',
             'temperature_measured', 'current_load', 'voltage_load', 'time']
    train_dataset = capacity[attribs]
    sc = MinMaxScaler(feature_range=(0,1)) # = (num-min)/(max-min)
    train_dataset = sc.fit_transform(train_dataset) # issue：not based on Rated
    # print(train_dataset.shape)
    # print(soh.shape)
    attribs_scaled = pd.DataFrame(data=train_dataset,columns=attribs)
    return  pd.concat([capacity['cycle'], attribs_scaled, soh], axis=1)

def data_loader(battery_id,seq_len,input_size):
    dataset = norm_data(battery_id)
    data_set_train=dataset[['cycle','SOH']].values
    x_train=[]
    y_train=[]
    #take the last 10t to predict 10t+1
    for i in range(seq_len,len(data_set_train)):
        x_train.append(data_set_train[i-10:i,1])
        y_train.append(data_set_train[i,1])
    x_train = np.array(x_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],input_size)) #(batch,seq_len,input_size)
    return x_train,y_train


