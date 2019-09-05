import os
import numpy as np
import pickle
from familyGan.stylegan_encoder import dnnlib
import familyGan.stylegan_encoder.dnnlib.tflib as tflib
from familyGan.stylegan_encoder import config
from familyGan.stylegan_encoder.encoder.generator_model import Generator
import sys

sys.modules['dnnlib'] = dnnlib
sys.modules['tflib'] = tflib

FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = f'{os.path.dirname(os.path.dirname(FAMILYGAN_DIR_PATH))}/familyGan_data/TSKinFace_Data'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# tflib.init_tf()
# with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
#     generator_network, discriminator_network, Gs_network = pickle.load(f)
# generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

direction_path = os.path.join(FAMILYGAN_DIR_PATH, 'familyGan', 'stylegan_encoder', 'trained_directions')
gender_direction = np.load(f'{direction_path}/gender_direction.npy')
headPose_yaw_direction = np.load(f'{direction_path}/headPose_yaw_direction.npy')
headPose_roll_direction = np.load(f'{direction_path}/headPose_roll_direction.npy')
# headPose_pitch_direction = np.load(f'{direction_path}/headPose_pitch_direction.npy')  # no headPose_pitch data in training dataset
age_kid_direction = np.load(f'{direction_path}/age_young_direction.npy')
age_middle_direction = np.load(f'{direction_path}/age_middle_direction.npy')
age_young_direction = np.load(f'{direction_path}/age_young_direction.npy')
age_old_direction = np.load(f'{direction_path}/age_old_direction.npy')
glasses_direction = np.load(f'{direction_path}/glasses_direction.npy')
smile_direction = np.load(f'{direction_path}/smile_direction.npy')
anger_direction = np.load(f'{direction_path}/anger_direction.npy')
sadness_direction = np.load(f'{direction_path}/sadness_direction.npy')
contempt_direction = np.load(f'{direction_path}/contempt_direction.npy')
disgust_direction = np.load(f'{direction_path}/disgust_direction.npy')
fear_direction = np.load(f'{direction_path}/fear_direction.npy')
happiness_direction = np.load(f'{direction_path}/happiness_direction.npy')
neutral_direction = np.load(f'{direction_path}/neutral_direction.npy')
surprise_direction = np.load(f'{direction_path}/surprise_direction.npy')
eyeMakeup_direction = np.load(f'{direction_path}/eyeMakeup_direction.npy')
lipMakeup_direction = np.load(f'{direction_path}/lipMakeup_direction.npy')
beard_direction = np.load(f'{direction_path}/beard_direction.npy')
facialhair_direction = np.load(f'{direction_path}/facialhair_direction.npy')
moustache_direction = np.load(f'{direction_path}/moustache_direction.npy')
sideburns_direction = np.load(f'{direction_path}/sideburns_direction.npy')

all_directions = np.stack([gender_direction,
                           gender_direction,
                           headPose_yaw_direction,
                           headPose_roll_direction,
                           # headPose_pitch_directio
                           age_kid_direction,
                           age_middle_direction,
                           age_young_direction,
                           age_old_direction,
                           glasses_direction,
                           smile_direction,
                           anger_direction,
                           sadness_direction,
                           contempt_direction,
                           disgust_direction,
                           fear_direction,
                           happiness_direction,
                           neutral_direction,
                           surprise_direction,
                           eyeMakeup_direction,
                           lipMakeup_direction,
                           beard_direction,
                           facialhair_direction,
                           moustache_direction,
                           sideburns_direction])

MALE, FEMALE = 'm', 'f'
GENDERS = [MALE, FEMALE]
# FILE_FORMAT is $$$_N_N.png


aligned_path = f'{DATA_DIR_PATH}/aligned_images/'
generated_path = f'{DATA_DIR_PATH}/generated_images/'
latent_path = f'{DATA_DIR_PATH}/latent_representations/'
pkls_path = f'{DATA_DIR_PATH}/pkl_files/'
EMBEDDING_PATH = f'{DATA_DIR_PATH}/tmp_generator_out'
OUTPUT_FAKE_PATH = f'{DATA_DIR_PATH}/tmp_fake_children'
