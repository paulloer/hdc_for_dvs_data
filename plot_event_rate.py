import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (10, 10)
plt.style.use('dark_background')

# load recorded data
column_names = ['x', 'y', 'p', 't']
training_data, test_data = [], []
for i in range(10):
    filename = f'./numbers_dataset/n_{i}.csv'
    data = pd.read_csv(filename, sep=',')
    data.columns = column_names
    training_data.append(data.iloc[0:int(len(data)*0.8)])
    test_data.append(data.iloc[int(len(data)*0.8)+1:-1])

plt.plot(training_data[0]['t'], training_data[0]['p'])
plt.show()