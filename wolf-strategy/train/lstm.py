from glob import glob
import random

import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os
from tqdm import tqdm
import pickle
from keras.utils import plot_model
from utils import preprocess2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(y_true, y_pred, name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(name)
    plt.show()


class strategyLSTM(object):
    def __init__(self, config):
        self.config = config

    def myembedding(self, X_train, max_len=90):
        self.max_len = max_len
        self.pre_X_train = []
        self.index = 1
        self.word2vector = {}
        self.vector2word = {}
        print("make word2vec")
        for X in tqdm(X_train):
            for X_ in X:
                X_ = "".join(str(X_))
                if X_ not in self.word2vector:
                    self.word2vector[X_] = self.index
                    self.vector2word[self.index] = X_
                    self.index += 1
        pickle.dump(self.word2vector, open(self.config.OUTPUT_PATH +
                                           "/data/word2vector.pkl", mode="wb"))
        print("conv word")
        for X in tqdm(X_train):
            self.pre_tmp = []
            for X_ in X:
                X_ = "".join(str(X_))
                self.pre_tmp.append(self.word2vector[X_])
            while len(self.pre_tmp) != max_len:
                self.pre_tmp.append(0)
            self.pre_X_train.append(self.pre_tmp)
        return self.pre_X_train

    def init_data(self):
        print("load_file")
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
            pre1 = preprocess2.preprocess2()
            pre1.update(name)
            if pre1.is_finish:
                X_train.append(pre1.f_maps)
                Y_train_1.append(pre1.y_map1)
                Y_train_2.append(pre1.y_map2)
                Y_train_3.append(pre1.y_map3)
                Y_train_4.append(pre1.y_map4)
                Y_train_5.append(pre1.y_map5)
            else:
                print(name)
        print("start embedding")
        X_train = self.myembedding(X_train)
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
            self.X_train)*self.config.TEST_PAR_VALID)], self.X_train[int(len(self.X_train)*self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_1, self.Y_valid_1 = self.Y_train_1[:int(len(
            self.Y_train_1)*self.config.TEST_PAR_VALID)], self.Y_train_1[int(len(self.Y_train_1)*self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_2, self.Y_valid_2 = self.Y_train_2[:int(len(
            self.Y_train_2)*self.config.TEST_PAR_VALID)], self.Y_train_2[int(len(self.Y_train_2)*self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_3, self.Y_valid_3 = self.Y_train_3[:int(len(
            self.Y_train_3) * self.config.TEST_PAR_VALID)], self.Y_train_3[int(len(self.Y_train_3) * self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_4, self.Y_valid_4 = self.Y_train_4[:int(len(
            self.Y_train_4) * self.config.TEST_PAR_VALID)], self.Y_train_4[int(len(self.Y_train_4) * self.config.TEST_PAR_VALID) + 1:]

        self.Y_test_5, self.Y_valid_5 = self.Y_train_5[:int(len(
            self.Y_train_5)*self.config.TEST_PAR_VALID)], self.Y_train_5[int(len(self.Y_train_5)*self.config.TEST_PAR_VALID) + 1:]

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

    def build_network(self):

        main_input = Input(shape=(1, self.max_len,))
        lstm = LSTM(self.config.HIDDEN_UNITS,
                    return_sequences=False)(main_input)
        x = Dense(50, activation="relu")(lstm)
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

        opt = Adam(lr=self.config.LEARNING_RATE)
        model.compile(loss={
            "output_1": "categorical_crossentropy",
            "output_2": "categorical_crossentropy",
            "output_3": "categorical_crossentropy",
            "output_4": "categorical_crossentropy",
            "output_5": "categorical_crossentropy"
        },
            optimizer=opt,
            metrics=["accuracy"]
        )
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
        self.network = self.build_network()
        self.network.summary()
        self.train_iterations()

    def train_iterations(self):
        print("start fit")
        print(np.array(self.X_train).shape)
        self.X_train = np.array(self.X_train)
        self.X_train = self.X_train.reshape(-1, 1, self.max_len)
        self.X_test = np.array(self.X_test)
        self.X_test = self.X_test.reshape(-1, 1, self.max_len)
        print(np.array(self.X_train).shape)

        self.X_train = np.array(self.X_train)
        self.Y_train_1 = np.array(self.Y_train_1)
        self.Y_train_2 = np.array(self.Y_train_2)
        self.Y_train_3 = np.array(self.Y_train_3)
        self.Y_train_4 = np.array(self.Y_train_4)
        self.Y_train_5 = np.array(self.Y_train_5)
        self.X_test = np.array(self.X_test)
        self.Y_test_1 = np.array(self.Y_test_1)
        self.Y_test_2 = np.array(self.Y_test_2)
        self.Y_test_3 = np.array(self.Y_test_3)
        self.Y_test_4 = np.array(self.Y_test_4)
        self.Y_test_5 = np.array(self.Y_test_5)
        self.X_valid = np.array(self.X_valid)
        self.Y_valid_1 = np.array(self.Y_valid_1)
        self.Y_valid_2 = np.array(self.Y_valid_2)
        self.Y_valid_3 = np.array(self.Y_valid_3)
        self.Y_valid_4 = np.array(self.Y_valid_4)
        self.Y_valid_5 = np.array(self.Y_valid_5)
        history = self.network.fit(self.X_train,
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
                                       self.X_valid, [
                                           self.Y_valid_1, self.Y_valid_2, self.Y_valid_3, self.Y_valid_4, self.Y_valid_5]
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
