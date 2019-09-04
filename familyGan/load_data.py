import pickle as pkl
import random
from pathlib import Path
import os
from typing import List

import numpy as np
from tqdm import tqdm


def get_files_from_path(pathstring) -> List[str]:
    """
    Retrives file names from the folder and returns all pickle paths

    Args:
        pathstring: The folder path

    Returns:
        The all pickle paths
    """

    pkl_paths = []
    for file in Path(pathstring).glob("**/*.pkl"):
        pkl_paths.append(str(file))

    return pkl_paths


def load_data_for_training(folder_path, gender_filter=None) -> (np.array, np.array, np.array):
    print("Starting saved data loading")

    X_fathers_list, X_mothers_list, y_child_list = [], [], []
    X_fathers, X_mothers, y_child = None, None, None

    for filep in tqdm(get_files_from_path(folder_path)):
        if gender_filter is not None and os.path.basename(filep)[2] != gender_filter:
            continue
        with open(filep, 'rb') as f:
            (father_image, father_latent_f), (mother_image, mother_latent_f), (child_image, child_latent_f) = pkl.load(f)

            X_fathers_list.append(father_latent_f)
            X_mothers_list.append(mother_latent_f)
            y_child_list.append(child_latent_f)

    X_fathers = np.stack(X_fathers_list)
    X_mothers = np.stack(X_mothers_list)
    y_child = np.stack(y_child_list)

    print("finished data loading")

    return X_fathers, X_mothers, y_child

def load_false_triplets(X_fathers, X_mothers, y_child, example_amount) -> (np.array, np.array, np.array):
    """
    Expects output from load_data_for_training
    """
    y_child_perm_list = []
    ex_num = y_child.shape[0]
    for i in range(example_amount):
        new_i = i
        while new_i == i:
            new_i = random.randint(0, ex_num-1)

        y_child_perm_list.append(y_child[new_i, ::])
    y_child_perm = np.stack(y_child_perm_list)

    return X_fathers, X_mothers, y_child_perm