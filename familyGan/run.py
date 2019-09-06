from familyGan.config import EMBEDDING_PATH
from familyGan.load_data import merge_stylegan_outputs_to_triplet_pickles, load_data_for_training
from os.path import join as pjoin

if __name__ == '__main__':
    # region image2latent -
    aligned_path = pjoin(EMBEDDING_PATH, 'aligned_images')
    # generated_path = pjoin(EMBEDDING_PATH,'generated_images')
    latent_path = pjoin(EMBEDDING_PATH, 'latent_representations')

    output_path = merge_stylegan_outputs_to_triplet_pickles(aligned_path, latent_path)
    X_fathers, X_mothers, y_child = load_data_for_training()
    # TODO: load_data_for_deploy

    # endregion
    # regression

    # latent2image
