import torch
import torchhd
import numpy as np
import pandas as pd
import torchhd.tensors
from tqdm import tqdm
import pickle as pkl
import csv
from helper_functions import encode, generate_vector, plot_similarity_matrix, plot_similarity_matrix_spatial, get_training_test_data_metrics, augment_data_with_random_shifts_percentage
from helper_functions import TEMP_FRAME, DIM, K, POL_RANGE, DVS_HEIGHT, DVS_WIDTH, VSA_MODEL, DTYPE
torch.manual_seed(42)

def process_DVS_data():
    # load recorded data
    column_names = ['x', 'y', 'p', 't']
    training_data, test_data = [], []
    for i in range(10):
        filename = f'./numbers_dataset/n_{i}.csv'
        data = pd.read_csv(filename, sep=',')
        data.columns = column_names
        training_data.append(data.iloc[0:int(len(data)*0.8)])
        test_data.append(data.iloc[int(len(data)*0.8)+1:-1])

    # create seed HD vectors
    try:
        with open(f'./spatial_vectors_DIM_{DIM}_K_{K}_{VSA_MODEL}.pkl', 'rb') as file:
            spatial_vectors = pkl.load(file)
    except:
        spatial_vectors = np.array(DVS_HEIGHT*[DVS_WIDTH*[torch.zeros(DIM)]])
        for dvs_w in tqdm(range(int(DVS_WIDTH/K)), 'Spatial vectors', position=0):
            for dvs_h in range(int(DVS_HEIGHT/K)):
                corner_vectors = torchhd.random(4, DIM, vsa=VSA_MODEL, dtype=DTYPE)
                spatial_vectors[dvs_w*K][dvs_h*K] = corner_vectors[0]
                spatial_vectors[dvs_w*K][dvs_h*K+K-1] = corner_vectors[1]
                spatial_vectors[dvs_w*K+K-1][dvs_h*K] = corner_vectors[2]
                spatial_vectors[dvs_w*K+K-1][dvs_h*K+K-1] = corner_vectors[3]
                for win_w in range(K):
                    for win_h in range(K):
                        # row win_w, column win_h
                        if (win_w,win_h)!=(0,0) and (win_w,win_h)!=(0,K-1) and (win_w,win_h)!=(K-1,0) and (win_w,win_h)!=(K-1,K-1):
                            spatial_vectors[dvs_w*K+win_w][dvs_h*K+win_h] = generate_vector(corner_vectors, ratio_left_right=(K-1-win_h)/(K-1), ratio_top_bottom=(K-1-win_w)/(K-1))
        with open(f'./spatial_vectors_DIM_{DIM}_K_{K}_{VSA_MODEL}.pkl', 'wb') as file:
            pkl.dump(spatial_vectors, file)

    # check spatial vectors for orthogonality
    plot_similarity_matrix_spatial(spatial_vectors)

    temporal_vectors = torchhd.level(TEMP_FRAME, DIM, vsa=VSA_MODEL, dtype=DTYPE)
    polarity_vectors = torchhd.level(POL_RANGE, DIM, vsa=VSA_MODEL, dtype=DTYPE)

    # create class vectors from the training data
    try:
        with open(f'./class_vectors_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.pkl', 'rb') as file:
            class_vectors = pkl.load(file)
    except:
        class_vectors = len(training_data)*[[]]
        for n in tqdm(range(len(training_data)), 'Training', position=0):
            data = training_data[n]
            frame_vectors = encode(data, spatial_vectors, polarity_vectors, temporal_vectors)
            if len(frame_vectors)<2:
                class_vectors[n] = frame_vectors
            else:
                class_vectors[n] = torchhd.multibundle(torch.stack(frame_vectors))
        with open(f'./class_vectors_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.pkl', 'wb') as file:
            pkl.dump(class_vectors, file)

    plot_similarity_matrix(class_vectors)

    # test class vectors on test data
    try:
        with open(f'./test_frame_vectors_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.pkl', 'rb') as file:
            test_frame_vectors = pkl.load(file)
    except:
        test_frame_vectors = []
        for data in tqdm(test_data, 'Encoding test data', position=0):
            test_frame_vectors.append(encode(data, spatial_vectors, polarity_vectors, temporal_vectors))
        with open(f'./test_frame_vectors_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.pkl', 'wb') as file:
            pkl.dump(test_frame_vectors, file)

    n_samples = 0
    n_matches = 0
    for number in tqdm(range(len(class_vectors)), 'Inference', position=0):
        for sample in test_frame_vectors[number]:
            sim = [float(torchhd.cosine_similarity(sample, class_vectors[j])) for j in range(len(class_vectors))]
            if number == np.argmax(sim):
                n_matches += 1
            n_samples += 1

    inference_accuracy = n_matches/n_samples
    print(f"Inference Accuracy is {inference_accuracy}")

    with open(f'./inference_accuracy_DIM_{DIM}_{int(TEMP_FRAME/1e3)}_ms_K_{K}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([inference_accuracy])

if __name__ == '__main__':
    process_DVS_data()
