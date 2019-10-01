import argparse
from os.path import join as pjoin
import os
from PIL import Image

from config import URL_PRETRAINED_RESNET,PerceptParam
from familyGan.pipeline import align_image, image2latent, latent2image, image_list2latent_old, latent_list2image_list
import time
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from familyGan.stylegan_encoder.training.misc import load_pkl, save_pkl
from familyGan.stylegan_encoder import dnnlib
from familyGan.stylegan_encoder.encoder.perceptual_model import load_images

ITER = 450
LR = 2.  # not used ATM, uses on in perceptual_model.py
EXPERIMENT_NAME = f"resnet_notincluded_iter_{ITER}_lr_{LR}"
DATA_PATH = "../data"
RESULTS_PATH = "../results"
IM_PATH = pjoin(DATA_PATH, "toy_face.jpg")
IM2_PATH = pjoin(DATA_PATH, "toy_face2.jpg")
DLATENTS_CACHE = pjoin(RESULTS_PATH, "cache/dlatents")
VGG_EMBED_CACHE = pjoin(RESULTS_PATH, "cache/vgg_embeddings")
MODEL_CACHE = "../familyGan/cache"
RESNET_IMAGE_SIZE = 256

def run_single_image(perc_param=None):

    #  image2latent -
    # TODO: replace with pre-trained efficientnet for higher speed (use train_effnet.py)

    # predict initial dlatents with ResNet model
    resnet_path, __ = dnnlib.util.open_url_n_cache(URL_PRETRAINED_RESNET,cache_dir=MODEL_CACHE)
    ff_model = load_model(resnet_path)
    # init_dlatent = ff_model.predict(preprocess_input(np.array(im_aligned)))
    imgs = load_images([IM_PATH], image_size=RESNET_IMAGE_SIZE, align=True)
    init_dlatent = ff_model.predict(preprocess_input(imgs))

    start = time.time()
    print(f"started im2lat")
    _, aligned_latent = image2latent(imgs, iterations=ITER, init_dlatents = init_dlatent, perc_param=perc_param)
    end = time.time()
    print(f"took {end - start} sec")

    # latent2image
    save_pkl(aligned_latent, pjoin(DLATENTS_CACHE,f"toy_image_dlatent.pkl"))
    im_hat = latent2image(aligned_latent)
    im_hat.save(pjoin(RESULTS_PATH,f'{round(end - start,3)}_sec_'+ EXPERIMENT_NAME +'.png'))


def run_2_images():
    im = Image.open(IM_PATH)
    im2 = Image.open(IM2_PATH)

    #  image2latent -
    im_aligned = align_image(im)
    im2_aligned = align_image(im2)

    start = time.time()
    print(f"started im2lat")
    _, aligned_latent_list = image_list2latent_old([im_aligned, im2_aligned], iterations=ITER, learning_rate=LR)
    end = time.time()
    print(f"took {end - start} sec")

    # latent2image
    save_pkl(aligned_latent_list[0], pjoin(DLATENTS_CACHE, f"toy_image.pkl"))
    save_pkl(aligned_latent_list[1], pjoin(DLATENTS_CACHE, f"toy_image2.pkl"))

    im_hat_list = latent_list2image_list(aligned_latent_list)
    for i, im_hat in enumerate(im_hat_list):
        im_hat.save(pjoin(RESULTS_PATH, f'{round(end - start, 3)}_sec_{EXPERIMENT_NAME}_{i}.png'))


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(DLATENTS_CACHE, exist_ok=True)
    os.makedirs(MODEL_CACHE, exist_ok=True)

    perc_param = PerceptParam(lr=0.02
                              , decay_rate=0.9
                              , decay_steps=10
                              , image_size=256
                              , use_vgg_layer=9  # use_vgg_layer
                              , use_vgg_loss=0.4  # use_vgg_loss
                              , face_mask=False  # face_mask
                              , use_grabcut=True
                              , scale_mask=1.5
                              , mask_dir='masks'
                              , use_pixel_loss=1.5
                              , use_mssim_loss=100
                              , use_lpips_loss=100
                              , use_l1_penalty=1)

    perc_param.decay_steps *= 0.01 * ITER # precent from total iter

    # for i in range(1):
    #     run_2_images()
    run_single_image(perc_param)