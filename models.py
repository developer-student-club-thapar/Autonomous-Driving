from keras.applications import VGG16

from keras.models import Model, Sequential
from keras.layers import Lambda, Convolution2D, ELU, Dense, Flatten, Dropout, MaxPooling2D, Conv2D, ReLU, \
    BatchNormalization, MaxPooling3D, Conv3D


def comma_model(height, width, time_len=1):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model


def pretrained_vgg16(height, width):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))

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


def nvidia_model(input_shape):
    model = Sequential()

    model.add(Conv2D(16, (5, 5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(ReLU())
    model.add(Dense(1024))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(1))

    return model


def CNN_3D(input_shape):
    HEIGHT = 170
    WIDTH = 303

    model = Sequential()
    model.add(Conv3D(24, kernel_size=(2, 5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(36, kernel_size=(2, 5, 5), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(48, kernel_size=(2, 5, 5), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(64, kernel_size=(2, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='tanh'))

    return model
