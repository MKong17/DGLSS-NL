import numpy as np
import os
from tqdm import tqdm


np.set_printoptions(linewidth=200, threshold=np.inf) 

souce_folder_path_noise = './noise'
souce_folder_path_without = './without_noise'

output_folder_path_noise = 'noise_in_txt'
output_folder_path_without = 'without_noise_in_txt'

npy_files_1 = [os.path.join(souce_folder_path_noise, f) for f in os.listdir(souce_folder_path_noise) if f.endswith('.npy')]
npy_files_2 = [os.path.join(souce_folder_path_without, f) for f in os.listdir(souce_folder_path_without) if f.endswith('.npy')]

if not os.path.exists(output_folder_path_noise):
    os.makedirs(output_folder_path_noise)


if not os.path.exists(output_folder_path_without):
    os.makedirs(output_folder_path_without)


for file_name_1 in os.listdir(souce_folder_path_noise):   
    if file_name_1.endswith('.npy'):
        npy_files_1 = os.path.join(souce_folder_path_noise, file_name_1)
        labels = np.load(npy_files_1)
        labels = np.squeeze(labels)
        txt_filename = file_name_1.replace('.npy', '.txt')
        txt_path = os.path.join(output_folder_path_noise, txt_filename)

        with open(txt_path, 'w') as f:
            f.write(str(labels))


for file_name in os.listdir(souce_folder_path_without):
    if file_name.endswith('.npy'):
        npy_files_2 = os.path.join(souce_folder_path_without, file_name)
        labels = np.load(npy_files_2)
        txt_filename = file_name.replace('.npy', '.txt')
        txt_path = os.path.join(output_folder_path_without, txt_filename)

        with open(txt_path, 'w') as f:
            f.write(str(labels))



# try:
#     all_labels_array = np.concatenate(all_labels_noise, axis=0)
#     print(f"Success Shape:{all_labels_array.shape}")
# except ValueError:
#     print(f"Fail")

# used for single label

# output_path = 'output_labels_special_unnoisy_0.txt'
# with open(output_path, 'w') as f:
#     f.write(str(labels_1))

# print(f'this lable from {folder_path_1}:')
# print(labels_1)
# print(len(labels_1))

# labels_2 = np.load(folder_path_2)
# labels_2 = np.squeeze(labels_2)

# output_path = 'output_labels_noisy.txt'
# with open(output_path, 'w') as f:
#     f.write(str(labels_2))

# print(f'this lable from {folder_path_2}:')
# print(labels_2)
# print(len(labels_2))