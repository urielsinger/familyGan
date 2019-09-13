from os.path import join as pjoin
import os
from PIL import Image
from familyGan.pipeline import align_image, image2latent, latent2image
import time

from stylegan_encoder.training.misc import save_pkl, load_pkl

ITER = 450
LR = 2.  # not used ATM, uses on in perceptual_model.py
EXPERIMENT_NAME = f"init_iter_{ITER}_lr_{LR}"
IM_PATH = "../data/toy_face.jpg"
RESULTS_PATH = "../results"
DLATENTS_CACHE = pjoin(RESULTS_PATH, "cache/dlatents")
VGG_EMBED_CACHE = pjoin(RESULTS_PATH, "cache/vgg_embeddings")

if __name__ == '__main__':
    im = Image.open(IM_PATH)

    #  image2latent -
    im_aligned = align_image(im)
    init_dlatent = load_pkl(pjoin(DLATENTS_CACHE,f"toy_image_dlatent.pkl"))

    start = time.time()
    print(f"started im2lat")
    _, aligned_latent = image2latent(im_aligned, iterations=ITER, learning_rate=LR, init_dlatent = init_dlatent)
    end = time.time()
    print(f"took {end - start} sec")

    # latent2image
    save_pkl(aligned_latent, pjoin(DLATENTS_CACHE,f"toy_image.pkl"))
    im_hat = latent2image(aligned_latent)
    im_hat.save(pjoin(RESULTS_PATH,f'{round(end - start,3)}_sec_'+ EXPERIMENT_NAME +'.png'))

