import os
import random
import string
from os.path import join
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import pickle

from familyGan.config import URL_VGG_16
from familyGan import config
import familyGan.stylegan_encoder.config as stylgan_config
import familyGan.stylegan_encoder.dnnlib as dnnlib
from familyGan.models.simple_avarage import SimpleAverageModel
from familyGan.multiproc_util import parmap
from familyGan.stylegan_encoder.encoder.generator_model import Generator
from familyGan.stylegan_encoder.encoder.perceptual_model import PerceptualModel, PerceptualModelOld
from familyGan.stylegan_encoder.ffhq_dataset.face_alignment import image_align_from_image
from familyGan.stylegan_encoder.encode_images import split_to_batches

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from auto_tqdm import tqdm


def align_image(img, imsize: int = 256):
    face_landmarks = config.landmarks_detector.get_landmarks_from_image(np.array(img))
    aligned_img = image_align_from_image(img, face_landmarks)
    return aligned_img.resize((imsize, imsize))


def image2latent_old(img, iterations=1250, learning_rate=1., init_dlatent:Optional[np.ndarray]=None):
    if config.generator is None:
        config.init_generator()
    generator = Generator(config.Gs_network, 1)
    perceptual_model = PerceptualModelOld(256)
    perceptual_model.build_perceptual_model(generator.generated_image)

    perceptual_model.set_reference_images_from_image(np.array([np.array(img)]))
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations)
    with tqdm(total=iterations) as pbar:
        for iteration, loss in enumerate(op):
            pbar.set_description('Loss: %.2f' % loss)
            pbar.update()
            yield None, iteration/iterations

    print(f"final loss {loss}")
    generated_img = generator.generate_images()[0]
    latent = generator.get_dlatents()[0]

    yield generated_img, latent


def image2latent(img_list, iterations=1250, init_dlatents: Optional[np.ndarray] = None, args=None
                 , return_image: bool = False, is_aligned:bool=False) \
        -> Tuple[List[Optional[np.ndarray]], List[np.ndarray]]:
    """
    :return: sizes of (batch_size, img_height, img_width, 3), (batch_size, 18, 512)
    """
    batch_size = len(img_list)
    args = config.DEFAULT_PERC_PARAM if args is None else args
    config.init_generator(batch_size=batch_size)
    # generator = config.generator  # TODO: messes with parallel
    generator = Generator(config.Gs_network, batch_size=batch_size)
    generator.reset_dlatents()

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):
        with dnnlib.util.open_url(URL_VGG_16, cache_dir=stylgan_config.cache_dir) as f:
            perc_model = pickle.load(f)
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator)
    generated_images_list, generated_dlatents_list = [], []

    for images_batch in tqdm(split_to_batches(img_list, batch_size), total=len(img_list) // batch_size):
        names = [f"image_{n}" for n in range(len(images_batch))]
        # names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        perceptual_model.set_reference_images(images_batch, is_aligned=is_aligned)

        # FIXME: split also init_dlatents to batches
        if init_dlatents is not None:
            generator.set_dlatents(init_dlatents)

        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations)
        pbar = tqdm(op, position=0, leave=True, total=iterations)
        best_loss = None
        best_dlatent = None
        for loss_dict in pbar:
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                                                                     for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                best_loss = loss_dict["loss"]
                best_dlatent = generator.get_dlatents()
            generator.stochastic_clip_dlatents()

        print(" ".join(names), " Loss {:.4f}".format(best_loss))
        generator.set_dlatents(best_dlatent)

        generated_images_list = [generator.generate_images() if return_image else None]
        generated_dlatents_list += [generator.get_dlatents()]

    return generated_images_list, generated_dlatents_list


def predict(father_latent, mother_latent, **kwargs):
    model = SimpleAverageModel(**kwargs)
    child_latent = model.predict([father_latent], [mother_latent])

    return child_latent


def latent2image(latent: np.ndarray) -> Image.Image:
    config.init_generator()
    latent = latent.reshape((1, 18, 512))
    config.generator.set_dlatents(latent)
    img_array = config.generator.generate_images()[0]
    img = Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))


def latent_list2image_list(latent_arr: np.ndarray) -> List[Image.Image]:
    batch_size = len(latent_arr)
    # if config.generator is None:
    #     config.init_generator(batch_size=batch_size)
    config.generator.set_dlatents(latent_arr)
    img_arrays = config.generator.generate_images()
    img_list = [Image.fromarray(im, 'RGB').resize((256, 256)) for im in img_arrays]
    return img_list


def full_pipe(father, mother, **kwargs):
    father_latent, mother_latent = None, None
    father_hash = hash(tuple(np.array(father).flatten()))
    mother_hash = hash(tuple(np.array(mother).flatten()))

    cache_path = join(config.FAMILYGAN_DIR_PATH, 'cache', 'latent_space_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            image2latent_cache = pickle.load(handle)
    else:
        image2latent_cache = dict()

    if father_hash in image2latent_cache:
        father_latent = image2latent_cache[father_hash]
    if mother_hash in image2latent_cache:
        mother_latent = image2latent_cache[mother_hash]

    # to latent
    def parallel_tolatent(tpl):
        (i, aligned_image) = tpl
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        # config.Gs_network
        _, aligned_latent = image2latent_old(aligned_image)
        return aligned_latent

    print("starting latent extraction")
    # if father_latent is None and mother_latent is None:
    #     father_aligned = align_image(father)
    #     mother_aligned = align_image(mother)
    #     father_latent, mother_latent = list(
    #         parmap(parallel_tolatent, list(enumerate([father_aligned, mother_aligned]))))
    # elif father_latent is not None and mother_latent is None:
    #     mother_aligned = align_image(mother)
    #     mother_latent = list(parmap(parallel_tolatent, list(enumerate([mother_aligned]))))[0]
    # elif father_latent is None and mother_latent is not None:
    #     father_aligned = align_image(father)
    #     father_latent = list(parmap(parallel_tolatent, list(enumerate([father_aligned]))))[0]
    number_of_runs = 1*(father_latent is None) + 1*(mother_latent is None)
    if father_latent is None:
        father_aligned = align_image(father)
        gen = image2latent_old(father_aligned)
        _, father_latent = next(gen)
        while isinstance(father_latent, float):
            yield father_latent/number_of_runs, None
            _, father_latent = next(gen)
    yield 0.5, None
    if mother_latent is None:
        mother_aligned = align_image(mother)
        gen = image2latent_old(mother_aligned)
        _, mother_latent = next(gen)
        while isinstance(mother_latent, float):
            yield (number_of_runs-1)/2 + mother_latent/number_of_runs, None
            _, mother_latent = next(gen)
    yield 0.9999, None
    print("end latent extraction")

    # cache
    image2latent_cache[father_hash] = father_latent
    image2latent_cache[mother_hash] = mother_latent
    with open(cache_path, 'wb') as handle:
        pickle.dump(image2latent_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # model
    child_latent = predict(father_latent, mother_latent, **kwargs)
    with open(cache_path, 'wb') as handle:
        pickle.dump(image2latent_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to image
    child = latent2image(child_latent)

    yield 1.0, None
    yield child, child_latent

def integrate_with_web_get_child(path_father, path_mother):
    def randomString(stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(stringLength))

    father = Image.open(path_father)
    mother = Image.open(path_mother)

    kwargs = {'age_coef': 0, 'gender_coef': 0}

    gen = full_pipe(father, mother, **kwargs)
    child_image, child_latent = next(gen)
    while isinstance(child_image, float):
        yield child_image
        child_image, child_latent = next(gen)

    parent_path = os.path.dirname(os.path.dirname(path_father))
    random_string = randomString(30)
    child_image_path = join(parent_path, 'generated_files', random_string + '.png')
    child_latent_path = join(parent_path, 'generated_files', random_string + '.npy')
    child_image.save(child_image_path)
    np.save(child_latent_path, child_latent)

    yield random_string + '.png'

def latent_play(latent_vector, **coeffs):
    new_latent_vector = latent_vector.copy()

    for direction_type, coeff in coeffs.items():
        try:
            new_latent_vector[:8] = (new_latent_vector + coeff * eval(f'config.{direction_type}_direction'))[:8]
        except:
            print(f'no such direction {direction_type}')

    return new_latent_vector

def integrate_with_web_get_generated_child(child_latent_path, **kwargs):
    def randomString(stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(stringLength))

    child_latent = np.load(child_latent_path)

    new_child_latent = latent_play(child_latent, **kwargs)
    child = latent2image(new_child_latent)

    parent_path = os.path.dirname(child_latent_path)
    random_string = randomString(30)
    child_path = join(parent_path, random_string + '.png')
    child.save(child_path)
    return random_string + '.png'

#
# if __name__ == '__main__':
#     name = 'bibi'
#     father = Image.open('/data/home/morpheus/repositories/familyGan/custom_data/bibi.png')
#     mother = Image.open('/data/home/morpheus/repositories/familyGan/custom_data/sara.png')
#     child = full_pipe(father, mother)
#     child.save(f'/data/home/morpheus/repositories/familyGan/custom_data/{name}_child_coef{str(coef)}.png')
