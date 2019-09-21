import pickle

from sklearn.model_selection import train_test_split

from familyGan.config import EMBEDDING_PATH, OUTPUT_FAKE_PATH
from familyGan.load_data import merge_stylegan_outputs_to_triplet_pickles, load_data_for_training
from os.path import join as pjoin
from familyGan.data_handler import dataHandler
from familyGan.models.simple_avarage import SimpleAverageModel
import logging
import os

gender_filter = None  # None, FEMALE, MALE

model_config = dict(coef=-1.5)
latent_model = SimpleAverageModel(**model_config)

# model_config = dict(epochs=100, lr = 1, coef=-1.5)
# latent_model = RegressorAndDirection(**model_config)

data_handler = dataHandler()
TEST_RATIO = 0.3
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
    fathers_train, fathers_test, mothers_train, mothers_test \
        , child_train, child_test, file_list_train, file_list_test \
                = train_test_split(X_fathers, X_mothers, y_children, file_list,
                           test_size=TEST_RATIO, random_state=42)
    latent_model.fit(fathers_train, mothers_train, child_train)
    y_hat_children = latent_model.predict(fathers_train, mothers_train)
    # endregion

    # region LATENT2IMAGE
    children_fake = []
    for child_latent in y_hat_children:
        children_fake.append(data_handler.latent2image(child_latent))

    assert len(file_list_train) == len(children_fake)
    fake_path = pjoin(OUTPUT_FAKE_PATH, latent_model.__class__.__name__)
    if not os.path.isdir(OUTPUT_FAKE_PATH):
        os.mkdir(OUTPUT_FAKE_PATH + '/')
    if not os.path.isdir(fake_path):
        os.mkdir(fake_path + '/')
    for k, fakefile in enumerate(file_list_train):
        fake_filepath = pjoin(fake_path, os.path.basename(fakefile))
        with open(fake_filepath, 'wb') as f:
            pickle.dump((children_fake[k], y_hat_children[k]), f)

    # endregion
    # save the model
    model_path = f'/mnt/familyGan_data/TSKinFace_Data/models/'
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, f'{latent_model.__class__.__name__}.pkl'), 'wb') as f:
        pickle.dump(latent_model, f)
    logger.info("train_predict finished")
