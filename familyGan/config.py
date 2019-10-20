import os
from collections import namedtuple
from typing import Optional

import numpy as np
import pickle
from familyGan.stylegan_encoder import dnnlib
import familyGan.stylegan_encoder.dnnlib.tflib as tflib
from familyGan.stylegan_encoder import config
import sys

from familyGan.stylegan_encoder.align_images import unpack_bz2, LANDMARKS_MODEL_URL
from keras.utils import get_file
from familyGan.stylegan_encoder.ffhq_dataset.landmarks_detector import LandmarksDetector

from familyGan.stylegan_encoder.encoder.generator_model import Generator

sys.modules['dnnlib'] = dnnlib
sys.modules['tflib'] = tflib

FAMILYGAN_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = '/mnt/familyGan_data/TSKinFace_Data'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
URL_PRETRAINED_STYLEGAN = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
URL_PRETRAINED_RESNET = 'https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb'
URL_VGG_16 = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# generator_network, discriminator_network, Gs_network, generator = None, None, None, None
generator, Gs_network = None, None

PerceptParam = namedtuple('PerceptParam', ['image_size', 'decay_rate', 'decay_steps'
    , 'face_mask', 'use_grabcut', 'scale_mask', 'mask_dir', 'use_pixel_loss', 'use_l2_vgg_loss'
    , 'use_mssim_loss', 'use_lpips_loss', 'use_l1_penalty','use_vgg_layer','use_vgg_loss','lr'])

DEFAULT_PERC_PARAM = PerceptParam(lr=0.02
                              , decay_rate=0.9
                              , decay_steps=10 # precent from total iter
                              , image_size=256
                              , use_vgg_layer=9  # use_vgg_layer
                              , use_l2_vgg_loss = 0
                              , use_vgg_loss=0.4  # use_vgg_loss
                              , use_pixel_loss=1.5
                              , use_mssim_loss=100
                              , use_lpips_loss=100
                              , use_l1_penalty=1
                              , face_mask=False  # face_mask
                              , use_grabcut=True
                              , scale_mask=1.5
                              , mask_dir='masks'
                              )

def init_generator(batch_size=1):
    global generator, Gs_network

    generator, Gs_network = get_generator(batch_size)


def get_generator(batch_size=1):
    global generator, Gs_network
    if generator is not None:
        return generator, Gs_network
    tiled_dlatent, randomize_noise = False, False
    clipping_threshold = 2
    dlatent_avg = ''

    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
        del discriminator_network, generator_network
    generator = Generator(Gs_network, batch_size=batch_size, clipping_threshold=clipping_threshold,
                          tiled_dlatent=tiled_dlatent, randomize_noise=randomize_noise)
    if (dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(dlatent_avg))
    return generator, Gs_network


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
