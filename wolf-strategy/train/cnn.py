from glob import glob
import random

import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Flatten,
    Dense,
    Reshape)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.pooling import MaxPool2D, AveragePooling2D
import os
from tqdm import tqdm
import pickle
from keras.utils import plot_model
from utils import preprocess1


class strategyCNN(object):
    def __init__(self, config):
        self.config = config

    def init_data(self):
        file_names = glob(self.config.LOG_PATH)
        X_train = []
        Y_train = []
        for name in tqdm(file_names):
            pre1 = preprocess1.preprocess1()
            pre1.update(name)
            X_train.append(pre1.f_map)
            Y_train.append([
                pre1.y_map1,
                pre1.y_map2,
                pre1.y_map3,
                pre1.y_map4,
                pre1.y_map5])
        self.X_train, self.X_test = X_train[:int(len(
            X_train)*self.config.TRAIN_PAR_TEST)], X_train[int(len(X_train)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train, self.Y_test = Y_train[:int(len(
            Y_train)*self.config.TRAIN_PAR_TEST)], Y_train[int(len(Y_train)*self.config.TRAIN_PAR_TEST) + 1:]
        os.makedirs(self.config.OUTPUT_PATH+"/data", exist_ok=True)
        pickle.dump(self.X_train, open(self.config.OUTPUT_PATH +
                                       "/data/X_train.pkl", mode="wb"))
        pickle.dump(self.X_test, open(self.config.OUTPUT_PATH +
                                      "/data/X_test.pkl", mode="wb"))
        pickle.dump(self.Y_train, open(self.config.OUTPUT_PATH +
                                       "/data/y_train.pkl", mode="wb"))
        pickle.dump(self.Y_test, open(self.config.OUTPUT_PATH +
                                      "/data/y_test.pkl", mode="wb"))

    def build_network(self, depth=4, mkenerls=[64, 64, 64, 32], conv_conf=[2, 1], pooling_conf=["max", 2, 2], bn=False, dropout=True, rate=0.8, activation="relu", conf=[5, 5, 14], output_dim=5):
        mchannel, mheight, mwidth = conf
        input = Input(shape=(mchannel, mheight, mwidth))

        conv1 = Convolution2D(filters=mkenerls[0], kernel_size=(
            1, mwidth), strides=(1, 1), padding="valid")(input)
        activation1 = Activation("relu")(conv1)
        pool1 = MaxPool2D(pool_size=(2, 1), strides=(2, 1),
                          padding='same')(activation1)
        _k1, _n1 = map(int, pool1.shape[1:3])
        reshape_pool1 = Reshape((1, _k1, _n1))(pool1)

        conv2 = Convolution2D(filters=mkenerls[1], kernel_size=(
            1, _n1), strides=(1, 1), padding="valid")(reshape_pool1)
        activation2 = Activation("relu")(conv2)
        pool2 = MaxPool2D(pool_size=(2, 1), strides=(2, 1),
                          padding='same')(activation2)
        _k2, _n2 = map(int, pool2.shape[1:3])
        reshape_pool2 = Reshape((1, _k2, _n2))(pool2)
        _k3, _n3 = map(int, pool2.shape[1:3])

        conv4 = Convolution2D(filters=mkenerls[2], kernel_size=(
            1, _n3), strides=(1, 1), padding="valid")(reshape_pool2)
        activation4 = Activation("relu")(conv4)
        pool4 = MaxPool2D(pool_size=(2, 1), strides=(2, 1),
                          padding='same')(activation4)
        mFlatten = Flatten()(pool4)
        ms_output = Dense(output_dim=128)(mFlatten)
        msinput = Activation("sigmoid")(ms_output)
        output_1 = Dense(activation="softmax",
                         output_dim=output_dim, name="output_1")(msinput)
        output_2 = Dense(activation="softmax",
                         output_dim=output_dim, name="output_2")(msinput)
        output_3 = Dense(activation="softmax",
                         output_dim=output_dim, name="output_3")(msinput)
        output_4 = Dense(activation="softmax",
                         output_dim=output_dim, name="output_4")(msinput)
        output_5 = Dense(activation="softmax",
                         output_dim=output_dim, name="output_5")(msinput)
        model = Model(input=input, output=[
            output_1, output_2, output_3, output_4, output_5])
        return model

    def train(self):
        if self.config.INIT_DATA:
            self.X_train = pickle.load(
                open(self.config.DATA_PATH + "/X_train.pkl", "rb"))
            self.X_test = pickle.load(
                open(self.config.DATA_PATH + "/X_test.pkl", "rb"))
            self.Y_train = pickle.load(
                open(self.config.DATA_PATH + "/Y_train.pkl", "rb"))
            self.Y_test = pickle.load(
                open(self.config.DATA_PATH + "/Y_test.pkl", "rb"))
        else:
            self.init_data()
        print("finish init_data")
        self.network = self.build_network()
        self.network.summary()
        opt = Adam(lr=self.config.LEARNING_RATE)
        self.network.compile(loss={
            "output_1": "categorical_crossentropy",
            "output_2": "categorical_crossentropy",
            "output_3": "categorical_crossentropy",
            "output_4": "categorical_crossentropy",
            "output_5": "categorical_crossentropy"
        },
            optimizer=opt
        )
        self.train_iterations()

    def train_iterations(self):
        print("start fit")
        print(np.array(self.X_train).shape)
        print(np.array(self.Y_train).shape)
        history = self.network.fit(
            self.X_train,
            {
                "output_1": self.Y_train[0],
                "output_2": self.Y_train[1],
                "output_3": self.Y_train[2],
                "output_4": self.Y_train[3],
                "output_5": self.Y_train[4]
            },
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCH,
            validation_data=(
                self.X_test, self.Y_test
            ))
        with open(self.config.OUTPUT_PATH + "/History.txt", "w") as f:
            f.write(history)
        self.network.save_weights(self.config.OUTPUT_PATH + "/param.h5")
        self.y_pred = self.network.predict_classes(self.X_test)
        print(confusion_matrix(self.Y_test, self.y_pred))
