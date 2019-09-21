from os.path import join as pjoin
import os
from PIL import Image
from familyGan.pipeline import align_image, image2latent, latent2image, image_list2latent, latent_list2image_list
import time

from familyGan.stylegan_encoder.training.misc import load_pkl, save_pkl

ITER = 450
LR = 2.  # not used ATM, uses on in perceptual_model.py
EXPERIMENT_NAME = f"adam_iter_{ITER}_lr_{LR}"
IM_PATH = "../data/toy_face.jpg"
IM2_PATH = "../data/toy_face2.jpg"
RESULTS_PATH = "../results"
DLATENTS_CACHE = pjoin(RESULTS_PATH, "cache/dlatents")
VGG_EMBED_CACHE = pjoin(RESULTS_PATH, "cache/vgg_embeddings")

def run_single_image():
    im = Image.open(IM_PATH)

    #  image2latent -
    im_aligned = align_image(im)
    init_dlatent = load_pkl(pjoin(DLATENTS_CACHE,f"toy_image_dlatent.pkl"))

    start = time.time()
    print(f"started im2lat")
    _, aligned_latent = image2latent(im_aligned, iterations=ITER, learning_rate=LR, init_dlatent = None)
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
    init_dlatent = load_pkl(pjoin(DLATENTS_CACHE, f"toy_image_dlatent.pkl"))

    start = time.time()
    print(f"started im2lat")
    _, aligned_latent_list = image_list2latent([im_aligned, im2_aligned], iterations=ITER, learning_rate=LR,
                                               init_dlatent=None)  # init_dlatent)
    end = time.time()
    print(f"took {end - start} sec")

    # latent2image
    save_pkl(aligned_latent_list[0], pjoin(DLATENTS_CACHE, f"toy_image.pkl"))
    save_pkl(aligned_latent_list[1], pjoin(DLATENTS_CACHE, f"toy_image2.pkl"))

    im_hat_list = latent_list2image_list(aligned_latent_list)
    for i, im_hat in enumerate(im_hat_list):
        im_hat.save(pjoin(RESULTS_PATH, f'{round(end - start, 3)}_sec_{EXPERIMENT_NAME}_{i}.png'))


if __name__ == '__main__':
    # for i in range(1):
    #     run_2_images()
    run_single_image()