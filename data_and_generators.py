import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise

from tensorflow.keras.utils import Sequence
from augmentation import Automold as am
from augmentation import Helpers as hp

# The dataset is sampled at a high frame rate that causes dataset to be redundant. Use `rate` to specify good sampling rate.
# Too similar images are not good, nor too dissimilar. Set rate accordingly

fig = plt.figure(figsize=(17, 17))
columns = 4
rows = 8
start = 24520
rate = 4

HEIGHT = 160
WIDTH = 320

for j, i in enumerate(range(start, start + rows * columns * rate, rate)):
    img = cv2.imread(f'all/{i}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, j + 1)


def load_data(rate=4, action_file='actions.csv', angle_file='angles.csv', actions=None, val_size=0.2, test_size=0.1,
              scale=False, shuffle=True):
    df_angles = pd.read_csv(angle_file)
    df_angles['angle'] = df_angles['angle'].astype('float')

    if actions:
        df_actions = pd.read_csv(action_file)
        df_actions['action'] = df_actions['action'].astype('str')
        df = df_angles.merge(df_actions, on='filename', how='inner')
    else:
        df = df_angles

    df = df[::rate]

    scaler = StandardScaler()

    if scale:
        df[['angle']] = scaler.fit_transform(df[['angle']])

    if actions:
        if not (len(actions) == 1 and actions[0] == 'all'):
            df = df.loc[df['action'].isin(actions)]

    train_df, val_df = train_test_split(df, test_size=(val_size + test_size), random_state=42, shuffle=shuffle)
    val_df, test_df = train_test_split(val_df, test_size=test_size, random_state=42, shuffle=shuffle)

    df.reset_index(inplace=True)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    return df, train_df, val_df, test_df, scaler


def load_data_3D_CNN(rate=3):
    df_angles = pd.read_csv('angles.csv')
    df_angles['angle'] = df_angles['angle'].astype('float')

    df_actions = pd.read_csv('actions.csv')
    df_actions['action'] = df_actions['action'].astype('category')

    df = df_angles.merge(df_actions, on='filename', how='inner')

    df = df[::rate]

    total = len(df)
    train_size = int(total * 0.8)
    test_size = val_size = int(total * 0.1)

    train_df = df[:train_size]
    val_df = df[train_size: train_size + val_size]
    test_df = df[train_size + val_size: train_size + val_size + test_size]

    if len(df) != len(df_actions) != len(df_angles):
        print('WARNING: length of columns soed not match !')

    df.reset_index(inplace=True)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    return df, train_df, val_df, test_df


def anticlockwise_rotation(image):
    angle = random.randint(0, 15)
    return rotate(image, angle)


def clockwise_rotation(image):
    angle = random.randint(0, 15)
    return rotate(image, -angle)


def add_noise(image):
    return random_noise(image)


def shift_left(image):
    x_shift = random.randint(0, 30)
    y_shift = random.randint(0, 30)

    transform = AffineTransform(translation=(x_shift, y_shift))
    shifted = warp(image, transform, preserve_range=True)
    return shifted


def shift_right(image):
    x_shift = random.randint(0, 30)
    y_shift = random.randint(0, 30)

    transform = AffineTransform(translation=(-x_shift, -y_shift))
    shifted = warp(image, transform, preserve_range=True)
    return shifted


def augment(image, functions=[anticlockwise_rotation, clockwise_rotation, shift_left, shift_right, add_noise]):
    function = random.choice(functions)
    aug_img = function(image)
    return aug_img


aug_img = augment(img / 255)


class DataGenerator2D(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, img_paths, angles, actions, base_path, augmentation_rate=0.4,
                 to_fit=True, return_actions=False, batch_size=32, dim=(320, 160), shuffle=True, scale_image=True,
                 lower_augmentation_angle=-999, upper_augmentation_angle=999):

        self.img_paths = img_paths.copy()
        if actions:
            self.actions = actions.copy()
        self.angles = angles.copy()
        self.base_path = base_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()
        self.return_actions = return_actions
        self.augmentation_rate = augmentation_rate
        self.scale_image = scale_image
        self.upper_augmentation_angle = upper_augmentation_angle
        self.lower_augmentation_angle = lower_augmentation_angle

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(len(self.img_paths) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        current_indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        img_paths_temp = self.img_paths[current_indexes]

        # Generate data
        X, flipped_indexes, augmented_indexes = self._generate_X(img_paths_temp)

        if self.to_fit:
            y = self._generate_y(current_indexes, flipped_indexes, augmented_indexes)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle == True:
            indices = np.arange(len(self.img_paths))
            np.random.shuffle(indices)
            self.img_paths, self.angles = self.img_paths[indices], self.angles[indices]
            self.img_paths.reset_index(drop=True, inplace=True)
            self.angles.reset_index(drop=True, inplace=True)

    def _generate_X(self, img_paths_temp):
        """Generates data containing batch_size images
        :param img_paths_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = []
        augmented_indexes = []
        flipped_indexes = []

        # Generate data
        for idx, path in zip(img_paths_temp.index, img_paths_temp):
            # Store sample
            img, is_flipped, is_augmented = self._load_image(path, self.angles[idx])

            if is_flipped:
                flipped_indexes.append(idx)

            if is_augmented:
                augmented_indexes.append(idx)

            if self.return_actions:
                X.append(np.array([img, self.actions[idx]]))
            else:
                X.append(img)

        return np.array(X), flipped_indexes, augmented_indexes

    def _generate_y(self, current_indexes, flipped_indexes, augmented_indexes):


#rn: batch if masks

        y = self.angles.iloc[current_indexes].copy()
        for idx in flipped_indexes:
            y[idx] *= -1

        return y.values


    def _load_image(self, image_path, angle):
        is_augmented = False
        is_flipped = False

        img = cv2.imread(self.base_path + '/' + image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image augmentation using automould

        if (np.random.random() < self.augmentation_rate):
            img = cv2.resize(img, (1280, 720))
            img = am.augment_random(img, volume='same',
                                    aug_types=["add_shadow", "add_snow", "add_rain", "add_fog", "add_gravel",
                                               "add_sun_flare", "add_speed"])

        if (np.random.random() < (self.augmentation_rate + 0.1)) and (
                (angle >= self.lower_augmentation_angle) & (angle <= self.upper_augmentation_angle)):
            is_flipped = True
            img = np.flip(img, 1)

        img = cv2.resize(img, self.dim)

        if (np.random.random() < (self.augmentation_rate + 0.1)) and (
                (angle >= self.lower_augmentation_angle) & (angle <= self.upper_augmentation_angle)):
            is_augmented = True
            img = augment(img)

        if self.scale_image:
            img = img / 255.0

        return (img, is_flipped, is_augmented)

class DataGeneratorModified2D(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, img_paths, angles, actions, base_path, augmentation_rate=0.4,
                 to_fit=True, return_actions=False, depth=8, dim=(303, 170), shuffle=True, overlap=5):
        self.img_paths = img_paths.copy()
        self.actions = actions.copy()
        self.angles = angles.copy()
        self.base_path = base_path
        self.to_fit = to_fit
        self.depth = depth
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()
        self.return_actions = return_actions
        self.augmentation_rate = augmentation_rate
        self.overlap = overlap
        self.delta = self.depth - self.overlap
        self.sequences = []

        temp_indexes = list(range(len(self.img_paths)))
        for i in range(0, len(self.img_paths) - self.depth, self.depth - self.overlap):
            self.sequences.append(temp_indexes[i:i + depth])

        self.sequences = np.array(self.sequences)
        np.random.shuffle(self.sequences)

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil((len(self.img_paths) - self.depth) / (self.depth - self.overlap)))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        current_indexes = self.sequences[index]

        img_paths_temp = self.img_paths[current_indexes]

        # Generate data
        X, flipped_indexes, augmented_indexes = self._generate_X(img_paths_temp)

        if self.to_fit:
            y = self._generate_y(current_indexes, flipped_indexes, augmented_indexes)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.img_paths)

    def _generate_X(self, img_paths_temp):
        """Generates data containing batch_size images
        :param img_paths_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = []
        augmented_indexes = []
        flipped_indexes = []

        # Generate data
        for idx, path in zip(img_paths_temp.index, img_paths_temp):
            # Store sample
            img, is_flipped, is_augmented = self._load_image(path, self.angles[idx])
            if is_flipped:
                flipped_indexes.append(idx)
            if is_augmented:
                augmented_indexes.append(idx)

            if self.return_actions:
                X.append(np.array([img, self.actions[idx]]))
            else:
                X.append(img)

        return np.array(X), flipped_indexes, augmented_indexes

    def _generate_y(self, current_indexes, flipped_indexes, augmented_indexes):

        y = self.angles.iloc[current_indexes].copy()
        for idx in flipped_indexes:
            y[idx] *= -1

        return y

    def _load_image(self, image_path, angle):

        is_augmented = False
        is_flipped = False

        img = cv2.imread(self.base_path + '/' + image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280, 720))

        if np.random.random() < self.augmentation_rate:
            is_augmented = True
            img = am.augment_random(img, volume='same',
                                    aug_types=["add_shadow", "add_snow", "add_rain", "add_fog", "add_gravel",
                                               "add_sun_flare", "add_speed"])

        if np.random.random() < 0 and int(angle) != 0:
            is_flipped = True
            img = np.flip(img, 1)

        img = cv2.resize(img, self.dim)
        img = img / 255

        return (img, is_flipped, is_augmented)


class DataGenerator3D(Sequence):

    def __init__(self, img_paths, angles, actions, base_path, augmentation_rate=0,
                 to_fit=True, return_actions=False, batch_size=32, depth=32,
                 dim=(303, 170), shuffle=False, return_all_ys=False, overlap=4):

        self.generator2D = DataGeneratorModified2D(img_paths=img_paths, angles=angles, actions=actions,
                                                   base_path=base_path, augmentation_rate=augmentation_rate,
                                                   to_fit=to_fit, return_actions=return_actions,
                                                   dim=dim, shuffle=shuffle, overlap=overlap, depth=depth)
        self.to_fit = to_fit
        self.depth = depth
        self.batch_size = batch_size
        self.return_all_ys = return_all_ys
        self.overlap = overlap
        self.delta = self.depth - self.overlap

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return self.generator2D.__len__() // self.batch_size

    def __getitem__(self, index):

        current_indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        X = []
        y = []

        for i in current_indexes:
            if self.to_fit:
                X_temp, y_temp = self.generator2D.__getitem__(i)
                X.append(X_temp)
                if self.return_all_ys:
                    y.append(y_temp)
                else:
                    y.append(y_temp.iloc[-1])
            else:
                X_temp = self.generator2D.__getitem__(i)

        if self.to_fit:
            return np.array(X), np.array(y)
        else:
            return np.array(X)
