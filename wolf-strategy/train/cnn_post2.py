"""
Epoch 10/10
99920/99920 [==============================] - 3s 27us/step - loss: 1.6083 - output_1_loss: 0.2388 - output_2_loss: 0.3384 - output_3_loss: 0.3403 - output_4_loss: 0.3452 - output_5_loss: 0.3455 - output_1_acc: 0.9047 - output_2_acc: 0.8613 - output_3_acc: 0.8614 - output_4_acc: 0.8615 - output_5_acc: 0.8583 - val_loss: 1.6942 - val_output_1_loss: 0.2596 - val_output_2_loss: 0.3792 - val_output_3_loss: 0.3616 - val_output_4_loss: 0.3608 - val_output_5_loss: 0.3331 - val_output_1_acc: 0.8956 - val_output_2_acc: 0.8463 - val_output_3_acc: 0.8521 - val_output_4_acc: 0.8564 - val_output_5_acc: 0.8618
"""
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
    Reshape,
    Conv2D)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.pooling import MaxPool2D, AveragePooling2D
import os
from tqdm import tqdm
import pickle
from keras.utils import plot_model
from utils import preprocess1
from utils import preprocess3
from utils import preprocess4
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns


def evaluate(y_true, y_pred, name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(name)
    plt.show()


class strategyCNN_post2(object):
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
        file_names = file_names[:1000]
        for name in tqdm(file_names):
            # pre1 = preprocess1.preprocess1()
            pre1 = preprocess4.preprocess3()
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

        self.X_test, self.X_valid = self.X_test[:int(len(
            self.X_test)*self.config.TEST_PAR_VALID)], self.X_test[int(len(self.X_test)*self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_1, self.Y_valid_1 = self.Y_test_1[:int(len(
            self.Y_test_1)*self.config.TEST_PAR_VALID)], self.Y_test_1[int(len(self.Y_test_1)*self.config.TEST_PAR_VALID) + 1:]
        self.Y_test_2, self.Y_valid_2 = self.Y_test_2[:int(len(
            self.Y_test_2)*self.config.TEST_PAR_VALID)], self.Y_test_2[int(len(self.Y_test_2)*self.config.TEST_PAR_VALID) + 1:]
        self.Y_test_3, self.Y_valid_3 = self.Y_test_3[:int(len(
            self.Y_test_3)*self.config.TEST_PAR_VALID)], self.Y_test_3[int(len(self.Y_test_3)*self.config.TEST_PAR_VALID) + 1:]
        self.Y_test_4, self.Y_valid_4 = self.Y_test_4[:int(len(
            self.Y_test_4)*self.config.TEST_PAR_VALID)], self.Y_test_4[int(len(self.Y_test_4)*self.config.TEST_PAR_VALID) + 1:]
        self.Y_test_5, self.Y_valid_5 = self.Y_test_5[:int(len(
            self.Y_test_5)*self.config.TEST_PAR_VALID)], self.Y_test_5[int(len(self.Y_test_5)*self.config.TEST_PAR_VALID) + 1:]

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.X_valid = np.array(self.X_valid)
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
        self.Y_valid_1 = np.array(self.Y_valid_1)
        self.Y_valid_2 = np.array(self.Y_valid_2)
        self.Y_valid_3 = np.array(self.Y_valid_3)
        self.Y_valid_4 = np.array(self.Y_valid_4)
        self.Y_valid_5 = np.array(self.Y_valid_5)

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

        pickle.dump(self.X_valid, open(self.config.OUTPUT_PATH +
                                       "/data/X_valid.pkl", mode="wb"))
        pickle.dump(self.Y_valid_1, open(self.config.OUTPUT_PATH +
                                         "/data/Y_valid_1.pkl", mode="wb"))
        pickle.dump(self.Y_valid_2, open(self.config.OUTPUT_PATH +
                                         "/data/Y_valid_2.pkl", mode="wb"))
        pickle.dump(self.Y_valid_3, open(self.config.OUTPUT_PATH +
                                         "/data/Y_valid_3.pkl", mode="wb"))
        pickle.dump(self.Y_valid_4, open(self.config.OUTPUT_PATH +
                                         "/data/Y_valid_4.pkl", mode="wb"))
        pickle.dump(self.Y_valid_5, open(self.config.OUTPUT_PATH +
                                         "/data/Y_valid_5.pkl", mode="wb"))

    def build_network_dense_agent(self):
        main_input = Input(shape=(len(self.X_train[1]),))
        x = Dense(200, activation="relu")(main_input)
        x = Dense(100, input_dim=200, activation="relu")(x)
        x = Dense(50, input_dim=100, activation="relu")(x)
        output_1 = Dense(5, activation="softmax",
                         input_dim=50, name="output_1")(x)
        output_2 = Dense(5, activation="softmax",
                         input_dim=50, name="output_2")(x)
        output_3 = Dense(5, activation="softmax",
                         input_dim=50, name="output_3")(x)
        output_4 = Dense(5, activation="softmax",
                         input_dim=50, name="output_4")(x)
        output_5 = Dense(5, activation="softmax",
                         input_dim=50, name="output_5")(x)
        model = Model(input=main_input, output=[
            output_1, output_2, output_3, output_4, output_5])
        return model

    def build_network_dense(self):
        main_input = Input(shape=(len(self.X_train[1])+5,))
        x = Dense(200,
                  activation="relu")(main_input)
        x = Dense(100, input_dim=200,
                  activation="relu")(x)
        x = Dense(50, input_dim=100,
                  activation="relu")(x)
        output_1 = Dense(4, activation="softmax",
                         input_dim=50, name="output_1")(x)
        output_2 = Dense(4, activation="softmax",
                         input_dim=50, name="output_2")(x)
        output_3 = Dense(4, activation="softmax",
                         input_dim=50, name="output_3")(x)
        output_4 = Dense(4, activation="softmax",
                         input_dim=50, name="output_4")(x)
        output_5 = Dense(4, activation="softmax",
                         input_dim=50, name="output_5")(x)
        model = Model(input=main_input, output=[
            output_1, output_2, output_3, output_4, output_5])
        return model

    def train(self):
        if self.config.INIT_DATA:
            self.X_train = pickle.load(
                open(self.config.DATA_PATH + "/X_train.pkl", "rb"))
            self.X_test = pickle.load(
                open(self.config.DATA_PATH + "/X_test.pkl", "rb"))
            self.X_valid = pickle.load(
                open(self.config.DATA_PATH + "/X_valid.pkl", "rb"))
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
            self.Y_valid_1 = pickle.load(
                open(self.config.DATA_PATH + "/Y_valid_1.pkl", "rb"))
            self.Y_valid_2 = pickle.load(
                open(self.config.DATA_PATH + "/Y_valid_2.pkl", "rb"))
            self.Y_valid_3 = pickle.load(
                open(self.config.DATA_PATH + "/Y_valid_3.pkl", "rb"))
            self.Y_valid_4 = pickle.load(
                open(self.config.DATA_PATH + "/Y_valid_4.pkl", "rb"))
            self.Y_valid_5 = pickle.load(
                open(self.config.DATA_PATH + "/Y_valid_5.pkl", "rb"))

        else:
            self.init_data()
        print("finish init_data")
        self.X_train = np.array(self.X_train)
        if self.config.NETWORK == "CNN":
            self.network = self.build_network_cnn()
            self.X_test = np.array(self.X_test)
            self.X_valid = np.array(self.X_valid)
        elif self.config.NETWORK == "GCNN":
            self.X_test = np.array(self.X_test)
            self.X_valid = np.array(self.X_valid)
            self.network = self.build_network()
        else:
            self.X_test = np.array(self.X_test)
            self.X_train_ = []
            for self.X_t in self.X_train:
                X_tra = self.X_t.reshape(
                    1, self.X_t.shape[0]*self.X_t.shape[1]*self.X_t.shape[2]).astype("float32")[0]
                self.X_train_.append(X_tra)
            self.X_train_ = np.array(self.X_train_)
            self.X_test_ = []
            for self.X_t in self.X_test:
                X_tra = self.X_t.reshape(
                    1, self.X_t.shape[0]*self.X_t.shape[1]*self.X_t.shape[2]).astype("float32")[0]
                self.X_test_.append(X_tra)
            self.X_valid = np.array(self.X_valid)
            self.X_valid_ = []
            for self.X_t in self.X_valid:
                X_vali = self.X_t.reshape(
                    1, self.X_t.shape[0]*self.X_t.shape[1]*self.X_t.shape[2]).astype("float32")[0]
                self.X_valid_.append(X_vali)

            # self.X_test_ = np.array(self.X_test_)
            self.X_train = np.array(self.X_train_)
            self.X_test = np.array(self.X_test_)
            self.X_valid = np.array(self.X_valid_)
            self.network = self.build_network_dense()
        self.network.summary()
        opt = Adam(lr=self.config.LEARNING_RATE)
        self.network.compile(loss={
            "output_1": "categorical_crossentropy",
            "output_2": "categorical_crossentropy",
            "output_3": "categorical_crossentropy",
            "output_4": "categorical_crossentropy",
            "output_5": "categorical_crossentropy"
        },
            optimizer=opt,
            metrics=["accuracy"]
        )
        self.train_iterations()

    def train_iterations(self):
        if self.config.IS_FINISH_TRAIN:
            print("start test")
            self.network.load_weights(self.config.OUTPUT_PATH + "/param.h5")
            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = self.network.predict(
                self.X_test, batch_size=len(self.X_test))

            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = np.argmax(self.y_pred_1, axis=1), np.argmax(
                self.y_pred_2, axis=1), np.argmax(self.y_pred_3, axis=1), np.argmax(self.y_pred_4, axis=1), np.argmax(self.y_pred_5, axis=1)
            self.Y_test_1, self.Y_test_2, self.Y_test_3, self.Y_test_4, self.Y_test_5 = np.argmax(self.Y_test_1, axis=1), np.argmax(
                self.Y_test_2, axis=1), np.argmax(self.Y_test_3, axis=1), np.argmax(self.Y_test_4, axis=1), np.argmax(self.Y_test_5, axis=1)

            self.Y_valid_1, self.Y_valid_2, self.Y_valid_3, self.Y_valid_4, self.Y_valid_5 = np.argmax(self.Y_valid_1, axis=1), np.argmax(
                self.Y_valid_2, axis=1), np.argmax(self.Y_valid_3, axis=1), np.argmax(self.Y_valid_4, axis=1), np.argmax(self.Y_valid_5, axis=1)

            evaluate(self.Y_test_1, self.y_pred_1, "test1.png")
            evaluate(self.Y_test_2, self.y_pred_2, "test2.png")
            evaluate(self.Y_test_3, self.y_pred_3, "test3.png")
            evaluate(self.Y_test_4, self.y_pred_4, "test4.png")
            evaluate(self.Y_test_5, self.y_pred_5, "test5.png")

        else:
            print("start fit")

            agent_network = self.build_network_dense_agent()
            opt = Adam(lr=self.config.LEARNING_RATE)
            agent_network.compile(loss={
                "output_1": "categorical_crossentropy",
                "output_2": "categorical_crossentropy",
                "output_3": "categorical_crossentropy",
                "output_4": "categorical_crossentropy",
                "output_5": "categorical_crossentropy"},
                optimizer=opt,
                metrics=["accuracy"]
            )
            agent_network.load_weights("result-CNN-dense/param.h5")
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = agent_network.predict(
                self.X_train, batch_size=len(self.X_train))
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = np.argmax(y_pred_1, axis=1), np.argmax(
                y_pred_2, axis=1), np.argmax(y_pred_3, axis=1), np.argmax(y_pred_4, axis=1), np.argmax(y_pred_5, axis=1)
            self.X_train = np.concatenate(
                (self.X_train.T, [y_pred_1], [y_pred_2], [y_pred_3], [y_pred_4], [y_pred_5]), axis=0)
            self.X_train = self.X_train.T

            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = agent_network.predict(
                self.X_test, batch_size=len(self.X_test))
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = np.argmax(y_pred_1, axis=1), np.argmax(
                y_pred_2, axis=1), np.argmax(y_pred_3, axis=1), np.argmax(y_pred_4, axis=1), np.argmax(y_pred_5, axis=1)
            self.X_test = np.concatenate(
                (self.X_test.T, [y_pred_1], [y_pred_2], [y_pred_3], [y_pred_4], [y_pred_5]), axis=0)
            self.X_test = self.X_test.T

            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = agent_network.predict(
                self.X_valid, batch_size=len(self.X_valid))
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = np.argmax(y_pred_1, axis=1), np.argmax(
                y_pred_2, axis=1), np.argmax(y_pred_3, axis=1), np.argmax(y_pred_4, axis=1), np.argmax(y_pred_5, axis=1)
            self.X_valid = np.concatenate(
                (self.X_valid.T, [y_pred_1], [y_pred_2], [y_pred_3], [y_pred_4], [y_pred_5]), axis=0)
            self.X_valid = self.X_valid.T
            # kmeans_model = KMeans(n_clusters=5,random_state=10)
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
                    self.X_valid, [self.Y_valid_1, self.Y_valid_2,
                                   self.Y_valid_3, self.Y_valid_4, self.Y_valid_5]
                ))

            self.network.save_weights(self.config.OUTPUT_PATH + "/param.h5")

            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = self.network.predict(
                self.X_test, batch_size=len(self.X_test))

            self.y_pred_1, self.y_pred_2, self.y_pred_3, self.y_pred_4, self.y_pred_5 = np.argmax(self.y_pred_1, axis=1), np.argmax(
                self.y_pred_2, axis=1), np.argmax(self.y_pred_3, axis=1), np.argmax(self.y_pred_4, axis=1), np.argmax(self.y_pred_5, axis=1)
            self.Y_test_1, self.Y_test_2, self.Y_test_3, self.Y_test_4, self.Y_test_5 = np.argmax(
                self.Y_test_1, axis=1), np.argmax(self.Y_test_2, axis=1), np.argmax(self.Y_test_3, axis=1), np.argmax(self.Y_test_4, axis=1), np.argmax(self.Y_test_5, axis=1)

            evaluate(self.Y_test_1, self.y_pred_1,
                     self.config.OUTPUT_PATH + "/test1.png")
            evaluate(self.Y_test_2, self.y_pred_2,
                     self.config.OUTPUT_PATH + "/test2.png")
            evaluate(self.Y_test_3, self.y_pred_3,
                     self.config.OUTPUT_PATH + "/test3.png")
            evaluate(self.Y_test_4, self.y_pred_4,
                     self.config.OUTPUT_PATH + "/test4.png")
            evaluate(self.Y_test_5, self.y_pred_5,
                     self.config.OUTPUT_PATH + "/test5.png")
            pickle.dump(history, open(self.config.OUTPUT_PATH +
                                      "/history_" + self.config.NETWORK + ".pkl", mode="wb"))
