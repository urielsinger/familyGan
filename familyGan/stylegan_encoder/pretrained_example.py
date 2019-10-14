# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
from PIL import Image

import familyGan.stylegan_encoder.dnnlib as dnnlib
import familyGan.stylegan_encoder.dnnlib.tflib as tflib
import familyGan.stylegan_encoder.config as config
from familyGan.stylegan_encoder.training import misc



fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
rnd = np.random.RandomState(5)
num_classes = 10

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, 'example.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

def main_conditional():
    # Initialize TensorFlow
    tflib.init_tf()

    # Load pre-trained network
    dir = 'results/00004-sgan-cifar10-1gpu-cond/'
    fn = 'network-snapshot-010372.pkl'
    _G, _D, Gs = pickle.load(open(os.path.join(dir,fn), 'rb'))

    # Print network details
    Gs.print_layers()


    # rnd = np.random.RandomState(10)

    # Initialize conditioning
    conditioning = np.eye(num_classes)


    for i, rnd in enumerate([np.random.RandomState(i) for i in np.arange(20)]):

        # Pick latent vector.
        latents = rnd.randn(num_classes, Gs.input_shape[1])

        # Generate image.
        images = Gs.run(latents, conditioning, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = images.reshape(32*10, 32, 3)

        # Save image.
        png_filename = os.path.join(dir, 'example_{}.png'.format(i))
        PIL.Image.fromarray(images, 'RGB').save(png_filename)

def main_binary():
    # Initialize Tensorflow
    tflib.init_tf()

    # Load pre-trained network
    dir = 'results/00005-sgancelebahq-binary-1gpu-cond-wgangp/'
    dir = 'results/00006-sgancelebahq-binary-1gpu-cond-wgangp/'

    fn = 'network-snapshot-006926.pkl'
    _, _, Gs = pickle.load(open(os.path.join(dir,fn), 'rb'))

    # Print network details
    Gs.print_layers()

    # Create binary attributes
    # eyeglasses, male, black_hair, smiling, young

    classes = {
        '5_o_Clock_Shadow':         0,
        'Arched_Eyebrows':          0,
        'Attractive':               1,
        'Bags_Under_Eyes':          0,
        'Bald':                     0,
        'Bangs':                    0,
        'Big_Lips':                 0,
        'Big_Nose':                 0,
        'Black_Hair':               0,
        'Blond_Hair':               0,
        'Blurry':                   0,
        'Brown_Hair':               1,
        'Bushy_Eyebrows':           0,
        'Chubby':                   0,
        'Double_Chin':              0,
        'Eyeglasses':               0,
        'Goatee':                   0,
        'Gray_Hair':                0,
        'Heavy_Makeup':             1,
        'High_Cheekbones':          1,
        'Male':                     0,
        'Mouth_Slightly_Open':      1,
        'Mustache':                 0,
        'Narrow_Eyes':              0,
        'No_Beard':                 0,
        'Oval_Face':                1,
        'Pale_Skin':                0,
        'Pointy_Nose':              0,
        'Receding_Hairline':        0,
        'Rosy_Cheeks':              0,
        'Sideburns':                0,
        'Smiling':                  0,
        'Straight_Hair':            0,
        'Wavy_Hair':                1,
        'Wearing_Earrings':         0,
        'Wearing_Hat':              0,
        'Wearing_Lipstick':         1,
        'Wearing_Necklace':         0,
        'Wearing_Necktie':          0,
        'Young':                    1
    }


    print([attr for (attr,key) in classes.items() if key==1])



    binary = np.array(list(classes.values())).reshape(1,-1)


    for i, rnd in enumerate([np.random.RandomState(i) for i in np.arange(20)]):

        latent = rnd.randn(1, Gs.input_shape[1])

        image = Gs.run(latent, binary, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        image = image.reshape(256,256,3)

        png_filename = os.path.join(dir, 'examples/example{}.png'.format(i))
        PIL.Image.fromarray(image, 'RGB').save(png_filename)



def main_textual():
    # Initialize Tensorflow
    tflib.init_tf()

    dir = 'results/00015-sgancoco_train-1gpu-cond'
    fn = 'network-snapshot-025000.pkl'
    _, _, Gs = pickle.load(open(os.path.join(dir,fn), 'rb'))

    # Print network details
    Gs.print_layers()
    embeddings = np.load('datasets/coco_test/coco_test-rxx.labels')
    fns=np.load('datasets/coco_test/fns.npy')

    # Use only 1 description (instead of all 5, to compare to attnGAN)
    embeddings = embeddings[0::5]
    fns = fns[0::5]

    for i, rnd in enumerate([np.random.RandomState(i) for i in np.arange(embeddings.shape[0])]):

        latent = rnd.randn(1, Gs.input_shape[1])

        emb = embeddings[i].reshape(1,-1)

        image = Gs.run(latent, emb, truncation_psi=0.8, randomize_noise=True, output_transform=fmt)

        image = image.reshape(256,256,3)

        png_filename = os.path.join(dir, 'examples/{}.png'.format(fns[i]))

        image = Image.fromarray(image)
        image.save(png_filename)

if __name__ == "__main__":
    main()
    # main_conditional()
    # main_binary()
    # main_textual()
