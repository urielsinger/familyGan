import pickle as pkl
import random
from pathlib import Path
import os
from typing import List

import PIL
import numpy as np

from auto_tqdm import tqdm

from familyGan import config


def get_files_from_path(pathstring, filetype: str = 'pkl') -> List[str]:
    """
    Retrives file names from the folder and returns all pickle paths

    Args:
        pathstring: The folder path

    Returns:
        The all pickle paths
    """

    pkl_paths = []
    for file in Path(pathstring).glob(f"**/*.{filetype}"):
        pkl_paths.append(str(file))

    return pkl_paths


def load_aligned_image_latent(fname_no_type, aligned_path, latent_path):
    img = PIL.Image.open(os.path.join(aligned_path, f'{fname_no_type}.png'))
    latent_f = np.load(os.path.join(latent_path, f'{fname_no_type}.npy'))
    return img, latent_f


def verify_files_exist(aligned_path, latent_path, parent_fname):
    if (not os.path.isfile(os.path.join(aligned_path, f'{parent_fname}.png'))) or (
            not os.path.isfile(os.path.join(latent_path, f'{parent_fname}.npy'))):
        return False
    return True


def merge_stylegan_outputs_to_triplet_pickles(aligned_path=config.aligned_path,
                                              latent_path=config.latent_path):
    print("starting merge from folders")
    for filep in tqdm(get_files_from_path(latent_path, filetype='npy')):

        fname = os.path.basename(filep)

        fname_no_type = fname[:-4]
        family_con, ex_num, end = fname_no_type.split('-')
        child_type, child_num = end.split('_')
        triplet_pkl_fname = f"{family_con}-{ex_num}-{child_type}_{child_num}.pkl"
        triplet_pkl_fname_path = f"{config.pkls_path}{triplet_pkl_fname}"
        if os.path.exists(triplet_pkl_fname):
            continue  # already merged

        if child_type == "F" or child_type == "M":
            continue  # do not go over father/mothers 

        child_img, child_latent_f = load_aligned_image_latent(fname_no_type, aligned_path, latent_path)

        father_fname_no_type = f"{family_con}-{ex_num}-F_{child_num}"
        mother_fname_no_type = f"{family_con}-{ex_num}-M_{child_num}"

        if not verify_files_exist(aligned_path, latent_path, father_fname_no_type) or not \
                verify_files_exist(aligned_path, latent_path, mother_fname_no_type):
            print(f"failed working on {fname_no_type}")
            continue

        father_img, father_latent_f = load_aligned_image_latent(father_fname_no_type, aligned_path, latent_path)
        mother_img, mother_latent_f = load_aligned_image_latent(mother_fname_no_type, aligned_path, latent_path)

        with open(triplet_pkl_fname_path, 'wb') as f:
            pkl.dump(((father_img, father_latent_f), (mother_img, mother_latent_f), (child_img, child_latent_f)), f)
    print("done merge from folders")
    return config.pkls_path


def load_data_for_training(pkl_folder_path, gender_filter=None) -> (np.array, np.array, np.array, list):
    print("Starting saved data loading")

    X_fathers_list, X_mothers_list, y_child_list, file_list = [], [], [], []
    X_fathers, X_mothers, y_child = None, None, None

    for filep in tqdm(get_files_from_path(pkl_folder_path, 'pkl')):
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
            new_i = random.randint(0, ex_num - 1)

        y_child_perm_list.append(y_child[new_i, ::])
    y_child_perm = np.stack(y_child_perm_list)

    return X_fathers[:example_amount, ::], X_mothers[:example_amount, ::], y_child_perm


def load_family_triplet_pkls(pkl_folder_path, ex_num=None):
    triplet_pkls = []
    for i, filep in enumerate(tqdm(get_files_from_path(pkl_folder_path), desc="loading family photos:", total=ex_num)):
        if ex_num is not None and i == ex_num:
            break
        with open(filep, 'rb') as f:
            triplet_pkls.append(pkl.load(f))
        # (father_image, father_latent_f), (mother_image, mother_latent_f), (child_image, child_latent_f) = pkl.load(
        # f)

    return triplet_pkls