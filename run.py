import os
FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = f'{FAMILYGAN_DIR_PATH}/../familyGan_data/TSKinFace_Data/'
from familyGan.data_handler import dataHandler

data_reader = dataHandler()
for im in data_reader.load_from_path(DATA_DIR_PATH):
    im
