import os
from typing import Optional

import numpy as np
import pickle
from stylegan_encoder import dnnlib
import stylegan_encoder.dnnlib.tflib as tflib
from stylegan_encoder import config
import sys

from stylegan_encoder.align_images import unpack_bz2, LANDMARKS_MODEL_URL
from keras.utils import get_file
from stylegan_encoder.ffhq_dataset.landmarks_detector import LandmarksDetector

sys.modules['dnnlib'] = dnnlib
sys.modules['tflib'] = tflib
from stylegan_encoder.encoder.generator_model import Generator

FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = '/mnt/familyGan_data/TSKinFace_Data'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

generator_network, discriminator_network, Gs_network, generator = None, None, None, None


def init_generator(init_dlatent:Optional[np.ndarray]=None):
    global generator_network, discriminator_network, Gs_network, generator
    if generator is None:
        tflib.init_tf()
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
        generator = Generator(Gs_network, batch_size=1, randomize_noise=False, init_dlatent=init_dlatent)


landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)

direction_path = os.path.join(FAMILYGAN_DIR_PATH, 'stylegan_encoder', 'trained_directions')
gender_direction = np.load('{}/gender_direction.npy'.format(direction_path))
headPose_yaw_direction = np.load('{}/headPose_yaw_direction.npy'.format(direction_path))
headPose_roll_direction = np.load('{}/headPose_roll_direction.npy'.format(direction_path))
# headPose_pitch_direction = np.load(f'{direction_path}/headPose_pitch_direction.npy')  # no headPose_pitch data in training dataset
age_kid_direction = np.load('{}/age_young_direction.npy'.format(direction_path))
age_middle_direction = np.load('{}/age_middle_direction.npy'.format(direction_path))
age_young_direction = np.load('{}/age_young_direction.npy'.format(direction_path))
age_old_direction = np.load('{}/age_old_direction.npy'.format(direction_path))
glasses_direction = np.load('{}/glasses_direction.npy'.format(direction_path))
smile_direction = np.load('{}/smile_direction.npy'.format(direction_path))
anger_direction = np.load('{}/anger_direction.npy'.format(direction_path))
sadness_direction = np.load('{}/sadness_direction.npy'.format(direction_path))
contempt_direction = np.load('{}/contempt_direction.npy'.format(direction_path))
disgust_direction = np.load('{}/disgust_direction.npy'.format(direction_path))
fear_direction = np.load('{}/fear_direction.npy'.format(direction_path))
happiness_direction = np.load('{}/happiness_direction.npy'.format(direction_path))
neutral_direction = np.load('{}/neutral_direction.npy'.format(direction_path))
surprise_direction = np.load('{}/surprise_direction.npy'.format(direction_path))
eyeMakeup_direction = np.load('{}/eyeMakeup_direction.npy'.format(direction_path))
lipMakeup_direction = np.load('{}/lipMakeup_direction.npy'.format(direction_path))
beard_direction = np.load('{}/beard_direction.npy'.format(direction_path))
facialhair_direction = np.load('{}/facialhair_direction.npy'.format(direction_path))
moustache_direction = np.load('{}/moustache_direction.npy'.format(direction_path))
sideburns_direction = np.load('{}/sideburns_direction.npy'.format(direction_path))

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


aligned_path = '{}/aligned_images/'.format(DATA_DIR_PATH)
generated_path = '{}/generated_images/'.format(DATA_DIR_PATH)
latent_path = '{}/latent_representations/'.format(DATA_DIR_PATH)
pkls_path = '{}/pkl_files/'.format(DATA_DIR_PATH)
EMBEDDING_PATH = '{}/tmp_generator_out'.format(DATA_DIR_PATH)
OUTPUT_FAKE_PATH = '{}/tmp_fake_children'.format(DATA_DIR_PATH)
