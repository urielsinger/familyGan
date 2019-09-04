import os
FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = f'{FAMILYGAN_DIR_PATH}/../../familyGan_data/TSKinFace_Data/aligned_images2'
SRMODEL_PATH = fr'{FAMILYGAN_DIR_PATH}/faceSuperResolution/checkpoints/generator_checkpoint.ckpt'
from familyGan.data_handler import dataHandler
import torch
from familyGan.faceSuperResolution.model import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
g_checkpoint = torch.load(SRMODEL_PATH)
generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)

data_reader = dataHandler()
data_reader.corp_from_original(f'{FAMILYGAN_DIR_PATH}/../../familyGan_data/TSKinFace_Data/tmp1')
