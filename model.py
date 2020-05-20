import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skimage import io
from skimage.transform import rotate, AffineTransform, warp
from skimage import img_as_ubyte
from skimage.util import random_noise

from tensorflow.keras.utils import Sequence
from augmentation import Automold as am
from augmentation import Helpers  as hp


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Flatten, Activation, Flatten, Dropout, GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Nadam, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from time import time

# The dataset is sampled at a high frame rate that causes dataset to be redundant. Use `rate` to specify good sampling rate.
# Too similar images are not good, nor too dissimilar. Set rate accordingly

fig = plt.figure(figsize=(17, 17))
columns = 4
rows = 8
start = 24520
rate = 4

for j, i in enumerate(range(start, start + rows*columns*rate, rate)):
    img = cv2.imread(f'all/{i}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, j+1)

def load_data(rate=4, action_file='actions.csv', angle_file='angles.csv', actions=None, val_size=0.2, test_size=0.1, scale=False, shuffle=True):
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


# angle_file = '/content/uncompressed_data/interpolated.csv'
angle_file = 'angles.csv'

df, train_df, val_df, test_df, scaler = load_data(angle_file=angle_file, rate=2, actions=['all'], scale=False)  # Or actions=['stay_in_lane', 'etc'] or actions = ['all']

print(df)

HEIGHT = 240
WIDTH = 320

def anticlockwise_rotation(image):
    angle= random.randint(0,15)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,15)
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

aug_img = augment(img/255)



class DataGenerator2D(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, img_paths, angles, actions, base_path, augmentation_rate=0.4,
                 to_fit=True, return_actions=False, batch_size=32, dim=(303, 170), shuffle=True):
        """Initialization
        :param img_paths: list of all 'label' ids to use in the generator
        :param angles: list of angles
        :param actions: list of actions
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
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
        """Generates data containing batch_size masks
        :param img_paths_temp: list of label ids to load
        :return: batch if masks
        """
        y = self.angles.iloc[current_indexes].copy()
        for idx in flipped_indexes:
            y[idx] *= -1

        return y

    def _load_image(self, image_path, angle):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        is_augmented = False
        is_flipped = False

        img = cv2.imread(self.base_path + '/' + image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if (np.random.random() < self.augmentation_rate) and ((angle >= -1.0) & (angle <= 1.0)):
            img = cv2.resize(img, (1280, 720))
            img = am.augment_random(img, volume='same',
                                    aug_types=["add_shadow", "add_snow", "add_rain", "add_fog", "add_gravel",
                                               "add_sun_flare", "add_speed"])

        img = cv2.resize(img, self.dim)
        img = img / 255.0

        if (np.random.random() < (self.augmentation_rate + 0.1)) and ((angle >= -5.0) & (angle <= 5.0)):
            is_flipped = True
            img = np.flip(img, 1)

        if (np.random.random() < (self.augmentation_rate + 0.1)) and ((angle >= -5.0) & (angle <= 5.0)):
            is_augmented = True
            img = augment(img)

        return (img, is_flipped, is_augmented)


train_generator = DataGenerator2D(train_df['filename'], train_df['angle'], None,
                                  './all',augmentation_rate=0.5, dim=(WIDTH, HEIGHT), batch_size=32, shuffle=True)

def pretrained_VGG16():

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

    # Select Number of layers to freeze

    for layer in base_model.layers[:7]:
        layer.trainable = False

    x = base_model.output
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(.5)(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(.5)(x)
    x = ReLU()(x)
    x = Dense(64)(x)
    predictions = Dense(1)(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    return model

model = pretrained_VGG16()
model.summary()

optimizer = Nadam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
tensorboard = TensorBoard(log_dir="./drive/My Drive/Self Driving/logs/VGG16/{}".format(time()), histogram_freq=1, write_graph=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=30)