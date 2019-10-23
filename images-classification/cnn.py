#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import keras
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc


def read_images(path, img_rows, img_cols):
    images = list()
    imgs_list = list()

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        if not os.path.isfile(img_path):
            continue

        img_postfix = img_name.split('.')[1]
        if img_postfix not in ['jpg', 'png']:
            continue

        images.append(img_path)

    for img_path in images:
        img = misc.imread(img_path)

        img = misc.imresize(img, size=(img_rows, img_cols))
        img = img.astype(np.float32)
        img /= 255.
        img = np.reshape(img, newshape=(img_rows, img_cols, 3))
        imgs_list.append(img)

    return np.array(imgs_list)


class CNNClassifier(object):
    dataset_path = 'data'
    train_set_path = os.path.join(dataset_path, 'train')
    test_set_path = os.path.join(dataset_path, 'test')
    resources_path = 'resources'
    tmp_path = 'tmp'
    models_path = 'models'
    logs_path = 'logs'

    def __init__(self, batch_size=64, target_size=64, drop_out=0.5, epochs=100, train=True):
        self.batch_size = batch_size
        self.target_size = target_size
        self.drop_out = drop_out
        self.epochs = epochs

        if train:
            self.cnn_model = self._init_model()
            self.train_cnn()
        else:
            cnn_model_name = os.path.join(self.models_path, 'cnn')
            self.cnn_model = keras.models.load_model(cnn_model_name)

    def _init_model(self):
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(filters=64, kernel_size=(3, 3), input_shape=(self.target_size, self.target_size, 3),
                                    activation='relu'))
        cnn_model.add(MaxPool2D(pool_size=(2, 2)))
        cnn_model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(MaxPool2D(pool_size=(2, 2)))
        cnn_model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(MaxPool2D(pool_size=(2, 2)))

        cnn_model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(MaxPool2D(pool_size=(2, 2)))

        cnn_model.add(Flatten())
        cnn_model.add(Dropout(self.drop_out))

        cnn_model.add(Dense(units=128, activation='relu'))
        cnn_model.add(Dropout(self.drop_out))

        cnn_model.add(Dense(units=128, activation='relu'))
        cnn_model.add(Dropout(self.drop_out))

        cnn_model.add(Dense(units=4, activation='softmax'))
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cnn_model.summary()

        return cnn_model

    def train_cnn(self):
        train_data_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,
                                                  horizontal_flip=True)
        test_data_generator = ImageDataGenerator(rescale=1. / 255)

        train_set = train_data_generator.flow_from_directory(self.train_set_path, target_size=self.target_size,
                                                             batch_size=self.batch_size, class_mode='categorical')
        test_data = test_data_generator.flow_from_directory(self.test_set_path, target_size=self.target_size,
                                                            batch_size=self.batch_size, class_mode='categorical')

        weights_path = os.path.join(self.tmp_path, 'weights.hdf5')
        ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=False)
        tensorboard = TensorBoard(log_dir=self.logs_path, histogram_freq=0, batch_size=self.batch_size,
                                  write_grads=True, write_images=True)
        self.cnn_model.fit_generator(train_set, max_queue_size=10, workers=10, steps_per_epoch=8000 / self.batch_size,
                                     epochs=self.epochs, validation_data=test_data,
                                     validation_steps=2000 / self.batch_size, callbacks=[tensorboard])
        cnn_model_name = os.path.join(self.models_path, 'cnn')
        self.cnn_model.save(cnn_model_name)

    def classify(self):
        images = read_images(self.test_set_path, self.target_size, self.target_size)

        results = self.cnn_model.predict_classes(images)
        return results


if __name__ == '__main__':
    classifier = CNNClassifier(epochs=50)
    print(classifier.classify())
