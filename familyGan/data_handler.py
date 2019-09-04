import os
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from math import ceil

import config
from config import DATA_DIR_PATH, FAMILYGAN_DIR_PATH


class dataHandler:
    def __init__(self):
        self.path = DATA_DIR_PATH

    def create_all_faces_dir(self, output_name):
        output_path = join(self.path, output_name)
        for folder in os.listdir(join(self.path, 'TSKinFace_cropped')):
            cur_data_path = join(self.path, folder)
            for image_name in os.listdir(cur_data_path):
                image_path = join(cur_data_path, image_name)
                image = Image.open(image_path)
                image.save(join(output_path, image_name))

    def load_from_path(self, path, batch_size=None):
        image_names = os.listdir(path)
        num_images = len(image_names)
        batch_size = batch_size if batch_size is not None else num_images
        num_batches = ceil(num_images / batch_size)

        for num_batch in range(num_batches):
            images = []
            batch_image_names = image_names[num_batch * batch_size:(num_batch + 1) * batch_size]
            for image_name in batch_image_names:
                image_path = join(path, image_name)
                image = Image.open(image_path)
                images.append(image)
            yield images

    def corp_from_original(self, output_path):
        source_path = join(self.path, 'TSKinFace_source')
        df = []
        for subset in ['FMS', 'FMD', 'FMSD']:
            cur_data_path = join(source_path, subset + '_information', f'{subset}_FaceList_Last_combine.txt')
            df.append(
                pd.read_table(cur_data_path, header=None, delimiter=' ', names=['path', 'character', 'x', 'y', 'r']))
        df = pd.concat(df)
        df['path'] = df['path'].map(lambda path: path.replace('\\', '/'))
        df.head()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for _, sample in df.iterrows():
            image_path = os.path.join(output_path, sample['path'])
            image_name = sample['path'][sample['path'].find('/') + 1:sample['path'].find('.')] + '-' + sample[
                'character'] + '.png'
            image = Image.open(image_path)

            left = sample['x'] - sample['r']
            top = sample['y'] - sample['r']
            right = sample['x'] + sample['r']
            bottom = sample['y'] + sample['r']
            new_image = image.crop((left, top, right, bottom))
            new_image.resize((1024, 1024), Image.ANTIALIAS)

            new_image.save(join(output_path, image_name))

    def get_triplets(self, path, gender=None):
        image_names = os.listdir(path)
        df = pd.DataFrame(np.array([image_names]).T, columns=['image_names'])
        df['subset'] = df['image_names'].map(lambda image_name: image_name[:image_name.find('-')])
        df['sample_num'] = df['image_names'].map(lambda image_name: image_name[image_name.find('-') + 1:image_name.find(
            '-') + 1 + image_name[image_name.find('-') + 1:].find('-')])
        df['character'] = df['image_names'].map(
            lambda image_name: image_name[-image_name[::-1].find('-'):image_name.find('.')])

        samples = []
        for (subset, sample_num), sample_df in df.groupby(['subset', 'sample_num']):
            father_path = join(path, sample_df[sample_df['character'] == 'F']['image_names'].iloc[0])
            mother_path = join(path, sample_df[sample_df['character'] == 'M']['image_names'].iloc[0])
            if (subset == 'FMS' or subset == 'FMSD') and (gender in [None, 'male']):
                child_path = join(path, sample_df[sample_df['character'] == 'S']['image_names'].iloc[0])
                sample = {'father': father_path, 'mother': mother_path, 'child': child_path}
                samples.append(sample)
            elif (subset == 'FMD' or subset == 'FMSD') and (gender in [None, 'female']):
                child_path = join(path, sample_df[sample_df['character'] == 'D']['image_names'].iloc[0])
                sample = {'father': father_path, 'mother': mother_path, 'child': child_path}
                samples.append(sample)

        return samples

    def align_images(self):
        script_path = f'{FAMILYGAN_DIR_PATH}/familyGan/stylegan-encoder/align_images.py'
        aligned_path = f'{self.path}/aligned_images/'
        faces_path = f'{self.path}/all_faces/'

        if not os.path.exists(aligned_path):
            os.makedirs(aligned_path)
        if not os.path.exists(faces_path):
            os.makedirs(faces_path)

        os.system(f"{script_path} {faces_path} {aligned_path}")

    def image2latent(self):
        script_path = f'{FAMILYGAN_DIR_PATH}/familyGan/stylegan-encoder/encode_images.py'

        if not os.path.exists(config.generated_path):
            os.makedirs(config.generated_path)
        if not os.path.exists(config.latent_path):
            os.makedirs(config.latent_path)

        os.system(f"{script_path} {config.aligned_path} {config.generated_path} {config.latent_path}")

    def latent2image(self, latent_vector):
        latent_vector = latent_vector.reshape((1, 18, 512))
        config.generator.set_dlatents(latent_vector)
        img_array = config.generator.generate_images()[0]
        img = Image.fromarray(img_array, 'RGB')
        return img.resize((256, 256))

    def latent_play(self, latent_vector, **coeffs):
        new_latent_vector = latent_vector.copy()

        for direction_type, coeff in coeffs.items():
            try:
                new_latent_vector[:8] = (latent_vector + coeff * eval(f'config.{direction_type}_direction'))[:8]
            except:
                print(f'no such direction {direction_type}')

        return new_latent_vector
