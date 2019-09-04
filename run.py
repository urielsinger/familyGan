import os
FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = f'{FAMILYGAN_DIR_PATH}/../../familyGan_data/TSKinFace_Data/aligned_images2'
from familyGan.data_handler import dataHandler

data_reader = dataHandler()
# for imbatch in data_reader.load_from_path(DATA_DIR_PATH, batch_size=10):
#     imbatch
#

data_reader.corp_from_original(f'{FAMILYGAN_DIR_PATH}/../../familyGan_data/TSKinFace_Data/tmp1')
