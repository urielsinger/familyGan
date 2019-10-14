import argparse
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
from config import URL_PRETRAINED_RESNET, PerceptParam
from familyGan.pipeline import align_image, image2latent, latent2image, image_list2latent_old, latent_list2image_list
import time
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from familyGan.stylegan_encoder.training.misc import load_pkl, save_pkl
from familyGan.stylegan_encoder import dnnlib
from familyGan.stylegan_encoder.encoder.perceptual_model import load_images

# region Run Param
ITER = 250
LR = 2.  # not used ATM, uses on in perceptual_model.py

USE_RESNET_INIT = False
LOAD_CACHE_DLATENT = False
RESNET_IMAGE_SIZE = 256
PERC_PARAM = PerceptParam(lr=0.06  # 0.02
                          , decay_rate=0.9
                          , decay_steps=10 * 0.01 * ITER  # precent from total iter
                          , image_size=256
                          , use_vgg_layer=9  # use_vgg_layer
                          # if the use_..._loss == 0 -> we don't use this loss
                          , use_l2_vgg_loss = 0
                          , use_vgg_loss  =0.4  # 0.4  # use_vgg_loss
                          , use_pixel_loss=1.5  # 1.5
                          , use_mssim_loss=0  # 100
                          , use_lpips_loss=0  # 100
                          , use_l1_penalty=0  # 1
                          , face_mask=False  # face_mask
                          , use_grabcut=True
                          , scale_mask=1.5
                          , mask_dir='masks'
                          )
# endregion

# region Run Paths
DATA_PATH = "../data"
RESULTS_PATH = "../results"
IM1_PATH = pjoin(DATA_PATH, "toy_face.jpg")
IM2_PATH = pjoin(DATA_PATH, "toy_face2.jpg")
DLATENTS_CACHE = pjoin(RESULTS_PATH, "cache/dlatents")
VGG_EMBED_CACHE = pjoin(RESULTS_PATH, "cache/vgg_embeddings")
MODEL_CACHE = "../familyGan/cache"
EXPERIMENT_NAME = "loss_decision_iter_{iter}_lr_{lr}"


# endregion

def run_single_image(perc_param=None):
    #  image2latent -
    # TODO: replace with pre-trained efficientnet for higher speed (use train_effnet.py)
    impath = IM1_PATH
    imname = impath.split('/')[-1].split('.')[0]
    im = Image.open(impath)
    imgs = [align_image(im)]
    init_dlatent = None

    # predict initial dlatents with ResNet model
    if USE_RESNET_INIT:
        resnet_path, __ = dnnlib.util.open_url_n_cache(URL_PRETRAINED_RESNET, cache_dir=MODEL_CACHE)
        ff_model = load_model(resnet_path)
        align_resize_im = align_image(im).resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE))
        imgs_resnet_pp = preprocess_input(np.expand_dims(np.array(align_resize_im), 0))
        init_dlatent = ff_model.predict(imgs_resnet_pp)

    if LOAD_CACHE_DLATENT:
        init_dlatent = load_pkl(pjoin(DLATENTS_CACHE, f"{imname}_dlatent.pkl"))

    start = time.time()
    print(f"started im2lat")

    _, aligned_latent = image2latent(imgs, iterations=ITER, init_dlatents=init_dlatent, args=perc_param,
                                     is_aligned=True)  # with new perceptual model
    aligned_latent = aligned_latent[0]

    # _, aligned_latent = image_list2latent_old(imgs, learning_rate=LR, iterations=ITER, init_dlatents = init_dlatent)

    end = time.time()
    print(f"took {end - start} sec")

    # latent2image
    save_pkl(aligned_latent, pjoin(DLATENTS_CACHE, f"{imname}_dlatent.pkl"))
    im_hat = latent_list2image_list(aligned_latent)
    im_hat[0].save(pjoin(RESULTS_PATH, f'{round(end - start, 3)}_sec_'
                         + EXPERIMENT_NAME.format(iter=ITER, lr=perc_param.lr) + '.png'))

    # save resnet initialization
    if USE_RESNET_INIT:
        im_hat = latent_list2image_list(init_dlatent)
        im_hat[0].save(pjoin(RESULTS_PATH, f'resnet_{imname}_init.png'))


def run_2_images():
    im = Image.open(IM1_PATH)
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
        im_hat.save(
            pjoin(RESULTS_PATH, f'{round(end - start, 3)}_sec_{EXPERIMENT_NAME.format(iter=ITER, lr=LR)}_{i}.png'))


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(DLATENTS_CACHE, exist_ok=True)
    os.makedirs(MODEL_CACHE, exist_ok=True)

    perc_param = PERC_PARAM

    # for i in range(1):
    #     run_2_images()
    run_single_image(perc_param)
