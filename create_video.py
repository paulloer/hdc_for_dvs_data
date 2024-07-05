import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from process_DVS_data import TEMP_WINDOW
from matplotlib.animation import FFMpegWriter

plt.rcParams["figure.figsize"] = (10, 10)
plt.style.use('dark_background')

filename = './numbers_dataset/n_0.csv'
T = 100000

column_names = ['x', 'y', 'p', 't']
dvs_data = pd.read_csv(filename, sep=',')
dvs_data.columns = column_names

t_max = max(dvs_data['t'])
n_windows = int(np.floor(t_max / T))

fig, ax = plt.subplots()
ax.set_xlim(0, 320)
ax.set_ylim(0, 320)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
plt.margins(0)
plt.title(filename)

metadata = dict(title='Hand Gesture Video', artist='Matplotlib', comment='Hand gesture visualization')
writer = FFMpegWriter(fps=5, metadata=metadata)

with writer.saving(fig, filename + f'_{T}_us.mp4', 100):
    for n in tqdm(range(n_windows - 1), 'Windows'):
        window_data = dvs_data[(n * T <= dvs_data['t']) & (dvs_data['t'] < (n + 1) * T)].reset_index(drop=True)
        ax.clear()
        ax.scatter(window_data['x'], window_data['y'], s=1, c=window_data['p'], marker='s', cmap='cool')
        ax.text(160, -20, f'Time: {n*T/1e6} s', horizontalalignment='center', verticalalignment='center')
        ax.text(160, -30, f'Frame: {n}', horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, 320)
        ax.set_ylim(0, 320)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.margins(0)
        plt.title(filename)
        writer.grab_frame()
