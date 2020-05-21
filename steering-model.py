from time import time
import argparse

from keras import backend as K
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from data_and_generators import DataGenerator2D, load_data, load_data_3D_CNN, DataGenerator3D
from models import comma_model, pretrained_vgg16, nvidia_model, CNN_3D

import joblib


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def models(model_name):
    df, train_df, val_df, test_df, scaler = load_data(angle_file='angles.csv',
                                                      rate=2,
                                                      actions=['all'],
                                                      scale=False)

    if model_name == 'comma.ai':
        HEIGHT, WIDTH = 160, 320
        model = comma_model(height=HEIGHT, width=WIDTH, time_len=1)

        train_generator = DataGenerator2D(train_df['filename'],
                                          train_df['angle'],
                                          actions=None,
                                          base_path='./all',
                                          augmentation_rate=0.4,
                                          dim=(WIDTH, HEIGHT),
                                          batch_size=128,
                                          shuffle=True)

        val_generator = DataGenerator2D(val_df['filename'],
                                        val_df['angle'],
                                        actions=None,
                                        base_path='./all',
                                        augmentation_rate=0,
                                        dim=(WIDTH, HEIGHT),
                                        batch_size=128,
                                        shuffle=True)

        tensorboard = TensorBoard(log_dir="logs/Comma.ai-Steering-Model/{}".format(time()),
                                  histogram_freq=1,
                                  write_graph=True)

        filepath = "save_model/Comma.ai" + "Comma.ai-Steering-Model-" + "saved-model-2-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False)

    if model_name == 'nvidia-dave2':
        HEIGHT, WIDTH = 240, 320
        model = nvidia_model(input_shape=(HEIGHT, WIDTH, 3))

        train_generator = DataGenerator2D(train_df['filename'],
                                          train_df['angle'],
                                          actions=None,
                                          base_path='./all',
                                          augmentation_rate=0.4,
                                          dim=(WIDTH, HEIGHT),
                                          batch_size=128,
                                          shuffle=True)

        val_generator = DataGenerator2D(val_df['filename'],
                                        val_df['angle'],
                                        actions=None,
                                        base_path='./all',
                                        augmentation_rate=0,
                                        dim=(WIDTH, HEIGHT),
                                        batch_size=128,
                                        shuffle=True)

        tensorboard = TensorBoard(log_dir="logs/Nvidia-Dave2-Steering-Model/{}".format(time()),
                                  histogram_freq=1,
                                  write_graph=True)

        filepath = "save_model/Nvidia-Dave2" + "Nvidia-Dave2-Steering-Model-" + "saved-model-2-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False)

    if model_name == 'pretrained_vgg16':
        HEIGHT, WIDTH = 160, 320
        model = pretrained_vgg16(HEIGHT, WIDTH)

        scaler_filename = 'scaler.pkl'

        joblib.dump(scaler, scaler_filename)
        lower_augmentation_angle = float(scaler.fit_transform([[-5]]))
        upper_augmentation_angle = float(scaler.fit_transform([[5]]))

        train_generator = DataGenerator2D(train_df['filename'],
                                          train_df['angle'],
                                          None,
                                          base_path="./all",
                                          augmentation_rate=0.2,
                                          dim=(WIDTH, HEIGHT),
                                          batch_size=128,
                                          shuffle=True,
                                          scale_image=False,
                                          lower_augmentation_angle=lower_augmentation_angle,
                                          upper_augmentation_angle=upper_augmentation_angle)

        val_generator = DataGenerator2D(val_df['filename'],
                                        val_df['angle'],
                                        None,
                                        base_path="./all",
                                        augmentation_rate=0,
                                        dim=(WIDTH, HEIGHT),
                                        batch_size=128,
                                        shuffle=True,
                                        scale_image=False)

        tensorboard = TensorBoard(log_dir="logs/Pretrained-VGG16-Steering-Model/{}".format(time()),
                                  histogram_freq=1,
                                  write_graph=True)

        filepath = "save_model/Pretrained-VGG16" + "Pretrained-VGG16-Steering-Model-" + "saved-model-2-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False)

    if model_name == '3D-CNN' :
        df, train_df, val_df, test_df = load_data_3D_CNN(rate=3)

        HEIGHT, WIDTH, DEPTH = 170, 303, 16

        model = CNN_3D(input_shape=(DEPTH, HEIGHT, WIDTH, 3))

        train_generator = DataGenerator3D(img_paths=train_df['filename'],
        								  angles=train_df['angle'],
                                          actions=train_df['action'],
                                          base_path='./all',
                                          dim=(303, 170),
                                          depth=16,
                                          batch_size=16,
                                          overlap=4,
                                          augmentation_rate=0.4)

        val_generator = DataGenerator3D(img_paths=val_df['filename'],
                                        angles=val_df['angle'],
                                        actions=val_df['action'],
                                        base_path='./all',
                                        dim=(303, 170),
                                        depth=16,
                                        batch_size=16,
                                        overlap=4)

        tensorboard = TensorBoard(log_dir="logs/CNN-3D-Steering-Model/{}".format(time()),
                                  histogram_freq=1,
                                  write_graph=True)

        filepath = "save_model/CNN-3D" + "CNN-3D-Steering-Model-" + "saved-model-2-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False)

    if model_name == 'CNN_LSTM':
    	df, train_df, val_df, test_df = load_data(rate=1)

        HEIGHT, WIDTH = 170, 303
    	model = CNN_LSTM(input_shape=(HEIGHT,WIDTH,3))
    	train_generator = DataGenerator3D(img_paths=train_df['filename'],angles= train_df['angle'], 
    												actions=train_df['action'], 
    												base_path='./all', 
    												dim=(303, 170),
    												batch_size=32,
    												overlap=5,
    												depth = 10,
    												augmentation_rate=0
    												)
		val_generator = DataGenerator3D(img_paths=val_df['filename'],
    												angles= val_df['angle'], 
    												actions=val_df['action'], 
    												base_path='./all', 
    												dim=(303, 170),
    												batch_size=32,
    												overlap=5,
    												depth = 10
    												)

		tensorboard = TensorBoard(log_dir="logs/CNN-LSTM-Steering-Model/{}".format(time()),
                                  histogram_freq=1,
                                  write_graph=True)

        filepath = "save_model/CNN-LSTM" + "CNN-LSTM-Steering-Model-" + "saved-model-2-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False)



    model.summary()
    optimizer = Nadam(lr=1e-6,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=30)

    callbacks_list = [checkpoint, tensorboard, early_stopping]

    NUM_EPOCHS = 1

    model.compile(optimizer,
                  loss=root_mean_squared_error,
                  metrics=['mean_squared_error'])

    history = model.fit_generator(train_generator,
                                  epochs=NUM_EPOCHS,
                                  shuffle=True,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator)


    






if __name__ == '__main__':
    # Models : comma.ai, pretrained_vgg16, nvidia-dave2, 3D-CNN

    parser = argparse.ArgumentParser(description='Neural Network for Steering a Self Driving Car')

    parser.add_argument('-m', '--model', type=str, default='comma.ai',
                        help='Name of the Model to be used, Modes Names are : comma.ai, pretrained_vgg16, nvidia-dave2, 3D-CNN (default : comma.ai)')

    args = vars(parser.parse_args())

    models(args['model'])
