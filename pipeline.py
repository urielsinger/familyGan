import os
import random
import string
from os.path import join
import numpy as np
from PIL import Image
import pickle

import config
from familyGan.multiproc_util import parmap
from familyGan.stylegan_encoder.encoder.perceptual_model import PerceptualModel
from familyGan.stylegan_encoder.ffhq_dataset.face_alignment import image_align_from_image
from familyGan.stylegan_encoder.encoder.generator_model import Generator
from familyGan.models.simple_avarage import SimpleAverageModel

coef = -2

def align_image(img):
    face_landmarks = config.landmarks_detector.get_landmarks_from_image(np.array(img))
    aligned_img = image_align_from_image(img, face_landmarks)
    return aligned_img.resize((256, 256))


def image2latent(img, iterations=1000):
    config.init_generator()
    generator = Generator(config.Gs_network, 1)
    perceptual_model = PerceptualModel(256)
    perceptual_model.build_perceptual_model(generator.generated_image)

    perceptual_model.set_reference_images_from_image(np.array([np.array(img)]))
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations)
    for iteration in range(iterations):
        next(op)

    generated_img = generator.generate_images()[0]
    latent = generator.get_dlatents()[0]

    return generated_img, latent


def predict(father_latent, mother_latent):
    model = SimpleAverageModel(coef=coef)
    child_latent = model.predict(father_latent, mother_latent)

    return child_latent


def latent2image(latent):
    config.init_generator()
    latent = latent.reshape((1, 18, 512))
    config.generator.set_dlatents(latent)
    img_array = config.generator.generate_images()[0]
    img = Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))


def full_pipe(father, mother, specific=''):
    cache_path = join(config.FAMILYGAN_DIR_PATH, 'custom_data', specific)
    if specific != '' and os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            father_latent, mother_latent = pickle.load(handle)
    else:
        # align
        father_aligned = align_image(father)
        mother_aligned = align_image(mother)

        # to latent
        def paralel_tolatent(tpl):
            (i, aligned_image) = tpl
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            # config.Gs_network
            _, aligned_latent = image2latent(aligned_image)
            return aligned_latent

        print("starting latent extraction")
        father_latent, mother_latent = list(parmap(paralel_tolatent, list(enumerate([father_aligned, mother_aligned]))))
        print("end latent extraction")
        # _, father_latent = image2latent(father_aligned)
        # _, mother_latent = image2latent(mother_aligned)

        print(father_latent)
        if specific != '':
            with open(cache_path, 'wb') as handle:
                pickle.dump((father_latent, mother_latent), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # model
    child_latent = predict(father_latent, mother_latent)

    # to image
    child = latent2image(child_latent)

    return child

def integrate_with_web(path_father, path_mother):
    def randomString(stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(stringLength))

    father = Image.open(path_father)
    mother = Image.open(path_mother)

    child = full_pipe(father, mother)

    parent_path = os.path.dirname(path_father)
    random_string = randomString(30)
    child_path = join(parent_path, random_string + '.png')
    child.save(child_path)
    return random_string + '.png'



if __name__ == '__main__':
    specific = ''
    father = Image.open('/data/home/morpheus/repositories/familyGan/custom_data/kineret-F.png')
    mother = Image.open('/data/home/morpheus/repositories/familyGan/custom_data/kineret-M.png')
    child = full_pipe(father, mother, specific=specific)
    child.save(f'/data/home/morpheus/repositories/familyGan/custom_data/{specific}_child_coef{str(coef)}.png')
