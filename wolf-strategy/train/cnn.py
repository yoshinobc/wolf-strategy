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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random


def evaluate(y_true, y_pred, name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(name)
    plt.show()


def compare_TV(history):
    import matplotlib.pyplot as plt

    # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accracy Plt
    plt.plot(epochs, acc, 'bo', label='training acc')
    plt.plot(epochs, val_acc, 'b', label='validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    plt.savefig("valid_acc.png")
    plt.figure()

    # 2) Loss Plt
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.savefig("valid_loss.png")


class strategyCNN(object):
    def __init__(self, config):
        self.config = config

    def init_data(self):
        file_names = glob(self.config.LOG_PATH)
        X_train = []
        Y_train_1 = []
        Y_train_2 = []
        Y_train_3 = []
        Y_train_4 = []
        Y_train_5 = []
        random.seed(0)
        random.shuffle(file_names)
        for name in tqdm(file_names):
            pre1 = preprocess1.preprocess1()
            pre1.update(name)
            if pre1.is_finish:
                X_train.append(pre1.f_map)
                Y_train_1.append(pre1.y_map1)
                Y_train_2.append(pre1.y_map2)
                Y_train_3.append(pre1.y_map3)
                Y_train_4.append(pre1.y_map4)
                Y_train_5.append(pre1.y_map5)
        self.X_train, self.X_test = X_train[:int(len(
            X_train)*self.config.TRAIN_PAR_TEST)], X_train[int(len(X_train)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train_1, self.Y_test_1 = Y_train_1[:int(len(
            Y_train_1)*self.config.TRAIN_PAR_TEST)], Y_train_1[int(len(Y_train_1)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train_2, self.Y_test_2 = Y_train_2[:int(len(
            Y_train_2)*self.config.TRAIN_PAR_TEST)], Y_train_2[int(len(Y_train_2)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train_3, self.Y_test_3 = Y_train_3[:int(len(
            Y_train_3)*self.config.TRAIN_PAR_TEST)], Y_train_3[int(len(Y_train_3)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train_4, self.Y_test_4 = Y_train_4[:int(len(
            Y_train_4)*self.config.TRAIN_PAR_TEST)], Y_train_4[int(len(Y_train_4)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train_5, self.Y_test_5 = Y_train_5[:int(len(
            Y_train_5)*self.config.TRAIN_PAR_TEST)], Y_train_5[int(len(Y_train_5)*self.config.TRAIN_PAR_TEST) + 1:]
        os.makedirs(self.config.OUTPUT_PATH+"/data", exist_ok=True)
        pickle.dump(self.X_train, open(self.config.OUTPUT_PATH +
                                       "/data/X_train.pkl", mode="wb"))
        pickle.dump(self.X_test, open(self.config.OUTPUT_PATH +
                                      "/data/X_test.pkl", mode="wb"))
        pickle.dump(self.Y_train_1, open(self.config.OUTPUT_PATH +
                                         "/data/Y_train_1.pkl", mode="wb"))
        pickle.dump(self.Y_test_1, open(self.config.OUTPUT_PATH +
                                        "/data/Y_test_1.pkl", mode="wb"))
        pickle.dump(self.Y_train_2, open(self.config.OUTPUT_PATH +
                                         "/data/Y_train_2.pkl", mode="wb"))
        pickle.dump(self.Y_test_2, open(self.config.OUTPUT_PATH +
                                        "/data/Y_test_2.pkl", mode="wb"))
        pickle.dump(self.Y_train_3, open(self.config.OUTPUT_PATH +
                                         "/data/Y_train_3.pkl", mode="wb"))
        pickle.dump(self.Y_test_3, open(self.config.OUTPUT_PATH +
                                        "/data/Y_test_3.pkl", mode="wb"))
        pickle.dump(self.Y_train_4, open(self.config.OUTPUT_PATH +
                                         "/data/Y_train_4.pkl", mode="wb"))
        pickle.dump(self.Y_test_4, open(self.config.OUTPUT_PATH +
                                        "/data/Y_test_4.pkl", mode="wb"))
        pickle.dump(self.Y_train_5, open(self.config.OUTPUT_PATH +
                                         "/data/Y_train_5.pkl", mode="wb"))
        pickle.dump(self.Y_test_5, open(self.config.OUTPUT_PATH +
                                        "/data/Y_test_5.pkl", mode="wb"))

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
            self.Y_train_1 = pickle.load(
                open(self.config.DATA_PATH + "/Y_train_1.pkl", "rb"))
            self.Y_test_1 = pickle.load(
                open(self.config.DATA_PATH + "/Y_test_1.pkl", "rb"))
            self.Y_train_2 = pickle.load(
                open(self.config.DATA_PATH + "/Y_train_2.pkl", "rb"))
            self.Y_test_2 = pickle.load(
                open(self.config.DATA_PATH + "/Y_test_2.pkl", "rb"))
            self.Y_train_3 = pickle.load(
                open(self.config.DATA_PATH + "/Y_train_3.pkl", "rb"))
            self.Y_test_3 = pickle.load(
                open(self.config.DATA_PATH + "/Y_test_3.pkl", "rb"))
            self.Y_train_4 = pickle.load(
                open(self.config.DATA_PATH + "/Y_train_4.pkl", "rb"))
            self.Y_test_4 = pickle.load(
                open(self.config.DATA_PATH + "/Y_test_4.pkl", "rb"))
            self.Y_train_5 = pickle.load(
                open(self.config.DATA_PATH + "/Y_train_5.pkl", "rb"))
            self.Y_test_5 = pickle.load(
                open(self.config.DATA_PATH + "/Y_test_5.pkl", "rb"))
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
        if self.config.IS_FINISH_TRAIN:
            print("start test")
            self.X_test = np.array(self.X_test)
            self.Y_test_1 = np.array(self.Y_test_1)
            self.Y_test_2 = np.array(self.Y_test_2)
            self.Y_test_3 = np.array(self.Y_test_3)
            self.Y_test_4 = np.array(self.Y_test_4)
            self.Y_test_5 = np.array(self.Y_test_5)
            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = self.network.predict(
                self.X_test, batch_size=len(self.X_test))

            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = np.argmax(self.y_pred_1, axis=1), np.argmax(
                self.y_pred_1, axis=1), np.argmax(self.y_pred_3, axis=1), np.argmax(self.y_pred_4, axis=1), np.argmax(self.y_pred_5, axis=1)
            print(self.Y_test_2)
            self.Y_test_1, self.Y_test_2, self.Y_test_3, self.Y_test_4, self.Y_test_5 = np.argmax(self.Y_test_1, axis=1), np.argmax(
                self.Y_test_2, axis=1), np.argmax(self.Y_test_3, axis=1), np.argmax(self.Y_test_4, axis=1), np.argmax(self.Y_test_5, axis=1)
            print(self.Y_test_2)
            print(self.Y_test_1.shape, self.y_pred_1.shape)
            #print("test", self.Y_test_1[:20])
            #print("test2", self.Y_test_2[:100])
            #print("test3", self.Y_test_3[:20])
            #print("pred", self.y_pred_1[:20])
            #print("pred2", self.y_pred_2[:20])
            #print("pred3", self.y_pred_3[:20])
            evaluate(self.Y_test_1, self.y_pred_1, "test.png")
            #print(confusion_matrix(self.Y_test_1, self.y_pred_1))
            #print(confusion_matrix(self.Y_test_2, self.y_pred_2))
            #print(confusion_matrix(self.Y_test_3, self.y_pred_3))
            #print(confusion_matrix(self.Y_test_4, self.y_pred_4))
            #print(confusion_matrix(self.Y_test_5, self.y_pred_5))
            exit()
        print("start fit")
        print(np.array(self.X_train).shape)
        print(np.array(self.Y_train_1).shape)
        # print(np.array(self.Y_train_1).shape)
        print(np.array(self.X_train[:2]).shape)
        print(np.array(self.X_train[0]).shape)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train_1 = np.array(self.Y_train_1)
        self.Y_train_2 = np.array(self.Y_train_2)
        self.Y_train_3 = np.array(self.Y_train_3)
        self.Y_train_4 = np.array(self.Y_train_4)
        self.Y_train_5 = np.array(self.Y_train_5)
        self.Y_test_1 = np.array(self.Y_test_1)
        self.Y_test_2 = np.array(self.Y_test_2)
        self.Y_test_3 = np.array(self.Y_test_3)
        self.Y_test_4 = np.array(self.Y_test_4)
        self.Y_test_5 = np.array(self.Y_test_5)
        #print("output: ",self.network.predict(self.X_train[:2],batch_size=2))
        history = self.network.fit(
            self.X_train,
            {
                "output_1": self.Y_train_1,
                "output_2": self.Y_train_2,
                "output_3": self.Y_train_3,
                "output_4": self.Y_train_4,
                "output_5": self.Y_train_5
            },
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCH,
            validation_data=(
                self.X_test, [self.Y_test_1, self.Y_test_2,
                              self.Y_test_3, self.Y_test_4, self.Y_test_5]
            ))
        # with open(self.config.OUTPUT_PATH + "/History.txt", "w") as f:
        #    f.write(history)
        # compare_TV(history)
        # pickle.dump(history,open("history.pkl","rb"))
        self.network.save_weights(self.config.OUTPUT_PATH + "/param.h5")
        #self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = self.network.predict_classes(self.X_test)
        self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = self.network.predict(
            self.X_test, batch_size=len(self.X_test))
        # print(self.y_pred_1[:4])
        # print("1",self.y_pred_1)
        # print("2",self.y_pred_2)
        # print("3",self.y_pred_3)
        self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = np.argmax(self.y_pred_1, axis=1), np.argmax(
            self.y_pred_1, axis=1), np.argmax(self.y_pred_3, axis=1), np.argmax(self.y_pred_4, axis=1), np.argmax(self.y_pred_5, axis=1)
        self.Y_test_1, self.Y_test_2, self.Y_test_3, self.Y_test_4, self.Y_test_5 = np.argmax(self.Y_test_1, axis=1), np.argmax(
            self.Y_test_2, axis=1), np.argmax(self.Y_test_3, axis=1), np.argmax(self.Y_test_4, axis=1), np.argmax(self.Y_test_5, axis=1)
        print(self.Y_test_1.shape, self.y_pred_1.shape)
        print("test", self.Y_test_1[:20])
        print("pred", self.y_pred_1[:20])
        print("pred2", self.y_pred_2[:20])
        print("pred3", self.y_pred_3[:20])
        evaluate(self.Y_test_1, self.y_pred_1, "test1.png")
        evaluate(self.Y_test_2, self.y_pred_2, "test2.png")
        evaluate(self.Y_test_3, self.y_pred_3, "test3.png")
        evaluate(self.Y_test_4, self.y_pred_4, "test4.png")
        evaluate(self.Y_test_5, self.y_pred_5, "test5.png")
        #with open("history.pkl","wb") as f:
        #    pickle.dump(history, f)
        pickle.dump(history, open(self.config.OUTPUT_PATH +"/data/history.pkl", mode="wb"))
        print(confusion_matrix(self.Y_test_1, self.y_pred_1))
        print(confusion_matrix(self.Y_test_2, self.y_pred_2))
        print(confusion_matrix(self.Y_test_3, self.y_pred_3))
        print(confusion_matrix(self.Y_test_4, self.y_pred_4))
        print(confusion_matrix(self.Y_test_5, self.y_pred_5))
