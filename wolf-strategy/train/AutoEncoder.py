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
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
random.seed(11)


def evaluate(y_true, y_pred, name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(name)
    # plt.show()


class strategyAE(object):
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
        self.count_all = {}
        self.all_sample = {}
        self.all_calm = {}
        self.all_liar = {}
        self.all_repel = {}
        self.all_follow = {}
        random.shuffle(file_names)
        file_names = file_names[:1000]
        for name in tqdm(file_names):
            pre1 = preprocess1.preprocess1()
            pre1.update(name)
            if pre1.is_finish:
                for key, value in pre1.count_content.items():
                    if key in self.count_all:
                        self.count_all[key] += value
                    else:
                        self.count_all[key] = value
                for key, value in pre1.count_sample.items():
                    if key in self.all_sample:
                        self.all_sample[key] += value
                    else:
                        self.all_sample[key] = value
                for key, value in pre1.count_calm.items():
                    if key in self.all_calm:
                        self.all_calm[key] += value
                    else:
                        self.all_calm[key] = value
                for key, value in pre1.count_liar.items():
                    if key in self.all_liar:
                        self.all_liar[key] += value
                    else:
                        self.all_liar[key] = value
                for key, value in pre1.count_repel.items():
                    if key in self.all_repel:
                        self.all_repel[key] += value
                    else:
                        self.all_repel[key] = value
                for key, value in pre1.count_follow.items():
                    if key in self.all_follow:
                        self.all_follow[key] += value
                    else:
                        self.all_follow[key] = value
                X_train.append(pre1.f_map)
                Y_train_1.append(pre1.y_map1)
                Y_train_2.append(pre1.y_map2)
                Y_train_3.append(pre1.y_map3)
                Y_train_4.append(pre1.y_map4)
                Y_train_5.append(pre1.y_map5)
        print(self.count_all)
        print(self.all_sample)
        print(self.all_calm)
        print(self.all_liar)
        print(self.all_repel)
        print(self.all_follow)

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

    def build_network_dense(self):
        main_input = Input(shape=(len(self.X_train[1]),))
        x = Dense(200,
                  activation="relu")(main_input)
        x = Dense(100, input_dim=200,
                  activation="relu")(x)
        x = Dense(50, input_dim=100,
                  activation="relu")(x)
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

    def train(self):

        self.init_data()
        print("finish init_data")
        self.X_train = np.array(self.X_train)
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

        self.X_test_ = np.array(self.X_test_)
        self.X_train = np.array(self.X_train_)
        self.X_test = np.array(self.X_test_)
        self.X_valid = np.array(self.X_valid_)
        self.network_ae = self.build_network_ae()

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
        self.train_iterations_ae()
        self.train_iterations()

    def train_iterations(self):
        print("start fit")
        #kmeans_model = KMeans(n_clusters=5,random_state=10)

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
        df_history = pd.DataFrame(history.history)
        df_history.plot()
        df_history.to_csv(self.config.OUTPUT_PATH + "/loss.png")
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
