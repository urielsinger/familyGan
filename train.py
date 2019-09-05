import pickle

from config import EMBEDDING_PATH, MALE, FEMALE, OUTPUT_FAKE_PATH
from familyGan.load_data import merge_stylegan_outputs_to_triplet_pickles, load_data_for_training
from os.path import join as pjoin
from familyGan.data_handler import dataHandler
from familyGan.models.simple_avarage import SimpleAverageModel
import logging

gender_filter = None # None, FEMALE, MALE
model_config = dict(coef = 1)
latent_model = SimpleAverageModel(**model_config)
data_handler = dataHandler()
logger = logging.getLogger("train")

if __name__ == '__main__':
    logger.info("train_predict started")
    # region IMAGE2LATENT
    aligned_path = pjoin(EMBEDDING_PATH, 'aligned_images')
    latent_path = pjoin(EMBEDDING_PATH, 'latent_representations')

    output_path = merge_stylegan_outputs_to_triplet_pickles(aligned_path, latent_path)
    X_fathers, X_mothers, y_children, file_list = load_data_for_training(output_path, gender_filter)
    # endregion

    # region REGRESSION
    # TODO: add train test split
    mdl = latent_model.fit(X_fathers, X_mothers, y_children)
    y_hat_children = mdl.predict(X_fathers, X_mothers)
    # endregion

    # region LATENT2IMAGE
    children_fake = []
    for child_latent in y_hat_children:
        children_fake.append(data_handler.latent2image(child_latent))

    # TODO: save name of file to 'tmp_fake/model/'
    assert len(file_list) == len(children_fake)
    for k, fakefile in enumerate(file_list):
        fake_path = pjoin(OUTPUT_FAKE_PATH,latent_model.__class__.__name__,fakefile)
        with open(fake_path, 'wb') as f:
            pickle.dump( (children_fake[k], y_hat_children[k]) )

    # endregion

    logger.info("train_predict finished")
