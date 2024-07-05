import torch
import torchhd
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
torch.manual_seed(42)

# constants
DVS_WIDTH = 320
DVS_HEIGHT = 320
POL_RANGE = 2  # either 0 or 1
DIM = 1000
MOV_WIN = 1
TEMP_FRAME = 100000  # in microseconds
K = 32  # in pixels
VSA_MODEL = 'HRR'
DTYPE = torch.float32
assert DVS_HEIGHT/K % 1 == 0 and DVS_WIDTH/K % 1 == 0, "K has to be a divider of DVS dimensions"

def encode(data:pd.DataFrame, spatial_vectors, polarity_vectors, temporal_vectors):
    t_min = min(data['t'])
    t_max = max(data['t'])-TEMP_FRAME
    n_frames = int(np.floor(MOV_WIN*(t_max-t_min)/TEMP_FRAME))
    frame_vectors = []
    # for n in tqdm(range(n_frames-1), 'Encoding', position=1, leave=False):
    for n in tqdm(range(n_frames-1), 'Encoding'):
        frame_data = data[(t_min+n*TEMP_FRAME/MOV_WIN <= data['t']) & (data['t'] < (t_min+(n*TEMP_FRAME/MOV_WIN)+TEMP_FRAME))].reset_index(drop=True)
        event_vectors = []
        for event in range(len(frame_data)):
            x = frame_data['x'][event]
            y = frame_data['y'][event]
            p = frame_data['p'][event]
            t = frame_data['t'][event] % TEMP_FRAME
            spatial_vector = torchhd.HRRTensor(spatial_vectors[x][y])  # TODO: create general solution
            event_vectors.append(temporal_vectors[t].bind(spatial_vector.bind(polarity_vectors[p])))
            # event_vectors.append(temporal_vectors[t].bind(spatial_vector))
            # event_vectors.append(spatial_vector)
        if len(event_vectors)<2:
            frame_vectors.append(event_vectors)
        else:
            frame_vectors.append(torchhd.multibundle(torch.stack(event_vectors)))
    return frame_vectors

def generate_vector(vectors:list, ratio_left_right, ratio_top_bottom):
    assert len(vectors[0]) == len(vectors[1]) == len(vectors[2]) == len(vectors[3]), "Vectors must have the same length"
    num_values = len(vectors[0])
    
    # Number of values to take from each vector
    num_from_vectors = []
    num_from_vectors.append(round(num_values * ratio_left_right * ratio_top_bottom))
    num_from_vectors.append(round(num_values * (1-ratio_left_right) * ratio_top_bottom))
    num_from_vectors.append(round(num_values * ratio_left_right * (1-ratio_top_bottom)))
    num_from_vectors.append(round(num_values * (1-ratio_left_right) * (1-ratio_top_bottom)))
    if sum(num_from_vectors)<num_values:
        num_from_vectors[np.argmax(num_from_vectors)] += num_values-sum(num_from_vectors)
    if sum(num_from_vectors)>num_values:
        num_from_vectors[np.argmax(num_from_vectors)] -= sum(num_from_vectors)-num_values
    assert sum(num_from_vectors)==num_values, 'Error in rounding'

    # Indices to select values from
    indices = torch.randperm(num_values)
    indices_vectors = []
    indices_vectors.append(indices[ : num_from_vectors[0]])
    indices_vectors.append(indices[num_from_vectors[0] : num_from_vectors[0]+num_from_vectors[1]])
    indices_vectors.append(indices[num_from_vectors[0]+num_from_vectors[1] : num_from_vectors[0]+num_from_vectors[1]+num_from_vectors[2]])
    indices_vectors.append(indices[num_from_vectors[0]+num_from_vectors[1]+num_from_vectors[2] : ])

    # Construct return_vector
    return_vector = torch.zeros_like(vectors[0])
    return_vector[indices_vectors[0]] = vectors[0][indices_vectors[0]]
    return_vector[indices_vectors[1]] = vectors[1][indices_vectors[1]]
    return_vector[indices_vectors[2]] = vectors[2][indices_vectors[2]]
    return_vector[indices_vectors[3]] = vectors[3][indices_vectors[3]]

    return return_vector


def plot_similarity_matrix(hypervectors):
    # Convert the list of hypervectors to a tensor
    hv_tensor = torch.stack(hypervectors)

    # Compute the pairwise similarity distances
    n = len(hypervectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = torchhd.cosine_similarity(hv_tensor[i], hv_tensor[j])

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='magma')
    plt.title('similarity matrix of relative similarity distances')
    plt.xlabel('class vectors')
    plt.ylabel('class vectors')
    plt.axis('equal')
    plt.savefig(f'./similarity_matrix_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.pdf')
    # plt.show()
    plt.close()

    return similarity_matrix

def plot_similarity_matrix_spatial(spatial_vectors):
    # confusion matrix with top left vector of frame as reference
    similarity_matrix = np.array(DVS_HEIGHT*[DVS_WIDTH*[0.0]])
    for dvs_w in range(int(DVS_WIDTH/K)):
        for dvs_h in range(int(DVS_HEIGHT/K)):
            orthogonal_vector = spatial_vectors[dvs_w*K][dvs_h*K]
            for win_w in range(K):
                for win_h in range(K):
                    # row win_w, column win_h
                    similarity_matrix[dvs_w*K+win_w][dvs_h*K+win_h] = float(torchhd.cosine_similarity(spatial_vectors[dvs_w*K+win_w][dvs_h*K+win_h], orthogonal_vector))
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=False, cmap='magma')
    plt.title('similarity matrix w.r.t. top left vector of frame')
    plt.xlabel('spatial vector x')
    plt.ylabel('spatial vector y')
    xticks, yticks = 2*[np.linspace(0, 320, 17, dtype=int)]
    plt.xticks(xticks, labels=xticks)
    plt.yticks(yticks, labels=yticks)
    plt.axis('equal')
    plt.savefig(f'./similarity_matrix_spatial_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}_top_left.pdf')
    # plt.show()
    plt.close()

    # confusion matrix with random vector as reference
    similarity_matrix = np.array(DVS_HEIGHT*[DVS_WIDTH*[0.0]])
    orthogonal_vector = np.array(torchhd.random(1, DIM, vsa=VSA_MODEL, dtype=DTYPE))
    for dvs_w in range(int(DVS_WIDTH/K)):
        for dvs_h in range(int(DVS_HEIGHT/K)):
            for win_w in range(K):
                for win_h in range(K):
                    # row win_w, column win_h
                    similarity_matrix[dvs_w*K+win_w][dvs_h*K+win_h] = float(torchhd.cosine_similarity(spatial_vectors[dvs_w*K+win_w][dvs_h*K+win_h], orthogonal_vector))
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=False, cmap='magma')
    plt.title('similarity matrix w.r.t. one random vector')
    plt.xlabel('spatial vector x')
    plt.ylabel('spatial vector y')
    xticks, yticks = 2*[np.linspace(0, 320, 17, dtype=int)]
    plt.xticks(xticks, labels=xticks)
    plt.yticks(yticks, labels=yticks)
    plt.axis('equal')
    plt.savefig(f'./similarity_matrix_spatial_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}_random.pdf')
    # plt.show()
    plt.close()

    return similarity_matrix


def get_training_test_data_metrics(training_data, test_data):
    len_training_data = [len(training_data[i]) for i in range(len(training_data))]
    n_frames_training = []
    for i in range(len(training_data)):
        t_min = min(training_data[i]['t'])
        t_max = max(training_data[i]['t'])-TEMP_FRAME
        n_frames_training.append(int(np.floor(MOV_WIN*(t_max-t_min)/TEMP_FRAME)))

    len_test_data = [len(test_data[i]) for i in range(len(test_data))]
    n_frames_test = []
    for i in range(len(test_data)):
        t_min = min(test_data[i]['t'])
        t_max = max(test_data[i]['t'])-TEMP_FRAME
        n_frames_test.append(int(np.floor(MOV_WIN*(t_max-t_min)/TEMP_FRAME)))

    return [len_training_data, len_test_data], [n_frames_training, n_frames_test]


def augment_data_with_random_shifts_percentage(df, percentage, shift):
    # Calculate the total number of new samples to generate
    num_new_samples = int(len(df) * percentage)
    
    # Create a new DataFrame to store the augmented data
    augmented_df = df.copy()
    
    # Randomly select rows to duplicate
    rows_to_augment = df.sample(num_new_samples, replace=True)
    
    # Generate new samples
    for _, row in rows_to_augment.iterrows():
        # Generate random shifts for 'x' and 'y' using a Gaussian distribution
        x_shift = np.random.randint(shift)-int(shift/2)
        y_shift = np.random.randint(shift)-int(shift/2)
        
        # Create new row with the same 'p' and 't', but with shifted 'x' and 'y'
        new_row = row.copy()
        new_row['x'] += x_shift
        new_row['y'] += y_shift
        if new_row['x'] >= DVS_WIDTH:
            new_row['x'] = DVS_WIDTH-1
        if new_row['x'] < 0:
            new_row['x'] = 0
        if new_row['y'] >= DVS_HEIGHT:
            new_row['y'] = DVS_HEIGHT-1
        if new_row['y'] < 0:
            new_row['y'] = 0
        augmented_df = pd.concat([augmented_df, pd.DataFrame([new_row])], ignore_index=True)
    augmented_df = augmented_df.sort_values(by=['t'], ascending=True)
    augmented_df.reset_index(drop=True, inplace=True)
    
    return augmented_df
