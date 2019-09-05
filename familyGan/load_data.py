import pickle as pkl
import random
from pathlib import Path
import os
from typing import List

import PIL
import numpy as np

import config
from auto_tqdm import tqdm


def get_files_from_path(pathstring) -> List[str]:
    """
    Retrives file names from the folder and returns all pickle paths

    Args:
        pathstring: The folder path

    Returns:
        The all pickle paths
    """

    pkl_paths = []
    for file in Path(pathstring).glob("**/*.npy"):
        pkl_paths.append(str(file))

    return pkl_paths

def load_aligned_image_latent(fname_no_type, aligned_path, latent_path):
    img = PIL.Image.open(os.path.join(aligned_path, f'{fname_no_type}.png'))
    latent_f = np.load(os.path.join(latent_path, f'{fname_no_type}.npy'))
    return img, latent_f


def merge_stylegan_outputs_to_triplet_pickles(aligned_path=config.aligned_path, generated_path=config.generated_path,
                                              latent_path=config.latent_path):
    print("starting merge from folders")
    for filep in tqdm(get_files_from_path(latent_path)):
        fname = os.path.basename(filep)

        fname_no_type = fname[:-4]
        print(fname_no_type)
        family_con, ex_num, end = fname_no_type.split('-')
        child_type, child_num = end.split('_')

        if child_type == "F" or child_type == "M":
            continue  # do not go over father/mothers 

        child_img, child_latent_f = load_aligned_image_latent(fname_no_type, aligned_path, latent_path)

        father_fname_no_type = f"{family_con}-{ex_num}-F_{child_num}"
        father_img, father_latent_f = load_aligned_image_latent(father_fname_no_type, aligned_path, latent_path)

        mother_fname_no_type = f"{family_con}-{ex_num}-M_{child_num}"
        mother_img, mother_latent_f = load_aligned_image_latent(mother_fname_no_type, aligned_path, latent_path)

        triplet_pkl_fname = f"{family_con}-{ex_num}-{child_type}_{child_num}.pkl"
        with open(f"{config.pkls_path}/{triplet_pkl_fname}", 'wb') as f:
            pkl.dump(((father_img, father_latent_f), (mother_img, mother_latent_f), (child_img, child_latent_f)), f)
    print("done merge from folders")

def load_data_for_training(pkl_folder_path, gender_filter=None) -> (np.array, np.array, np.array, list):
    print("Starting saved data loading")

    X_fathers_list, X_mothers_list, y_child_list, file_list = [], [], [], []
    X_fathers, X_mothers, y_child = None, None, None

    for filep in tqdm(get_files_from_path(pkl_folder_path)):
        if gender_filter is not None and os.path.basename(filep)[2] != gender_filter:
            continue
        with open(filep, 'rb') as f:
            (father_image, father_latent_f), (mother_image, mother_latent_f), (child_image, child_latent_f) = pkl.load(
                f)

            X_fathers_list.append(father_latent_f)
            X_mothers_list.append(mother_latent_f)
            y_child_list.append(child_latent_f)
            file_list.append(filep)

    X_fathers = np.stack(X_fathers_list)
    X_mothers = np.stack(X_mothers_list)
    y_child = np.stack(y_child_list)

    print("finished data loading")

    return X_fathers, X_mothers, y_child, file_list

def load_data_for_deploy(folder_path, gender_filter=None) -> (np.array, np.array):
    print("Starting saved data loading")

    X_fathers_list, X_mothers_list = [], []
    X_fathers, X_mothers = None, None

    for filep in tqdm(get_files_from_path(folder_path)):
        if gender_filter is not None and os.path.basename(filep)[2] != gender_filter:
            continue
        with open(filep, 'rb') as f:
            (father_image, father_latent_f), (mother_image, mother_latent_f) = pkl.load(f)

            X_fathers_list.append(father_latent_f)
            X_mothers_list.append(mother_latent_f)

    X_fathers = np.stack(X_fathers_list)
    X_mothers = np.stack(X_mothers_list)

    print("finished data loading")

    return X_fathers, X_mothers

def load_false_triplets(X_fathers, X_mothers, y_child, example_amount) -> (np.array, np.array, np.array):
    """
    Expects output from load_data_for_training

    returns new children for existing fathers,mothers (in the same order)
    """
    y_child_perm_list = []
    ex_num = y_child.shape[0]
    assert example_amount <= ex_num, "load_false expect a smaller number than the number of triplets"
    for i in range(example_amount):
        new_i = i
        while new_i == i:
            new_i = random.randint(0, ex_num-1)

        y_child_perm_list.append(y_child[new_i, ::])
    y_child_perm = np.stack(y_child_perm_list)

    return X_fathers[:example_amount, ::], X_mothers[:example_amount, ::], y_child_perm
