import os
import numpy as np
import pickle
from familyGan.stylegan_encoder import dnnlib
import familyGan.stylegan_encoder.dnnlib.tflib as tflib
from familyGan.stylegan_encoder import config
from familyGan.stylegan_encoder.encoder.generator_model import Generator

FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = f'{FAMILYGAN_DIR_PATH}/../familyGan_data/TSKinFace_Data/'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

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
