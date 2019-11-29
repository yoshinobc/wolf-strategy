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


class strategyLSTM(object):
    def __init__(self, config):
        self.config = config

    def myembedding(self, X_train):
        index = 0
        self.word2vector = {}
        self.vector2word = {}
        for X in X_train:
            if X not in self.word2vector:
                self.word2vector[X] = index
                self.vector2word[index] = X
                index += 1
        print(self.word2vector)

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
            pre1 = preprocess2.preprocess2()
            pre1.update(name)
            if pre1.is_finish:
                X_train.append(pre1.f_maps)
                Y_train_1.append(pre1.y_map1)
                Y_train_2.append(pre1.y_map2)
                Y_train_3.append(pre1.y_map3)
                Y_train_4.append(pre1.y_map4)
                Y_train_5.append(pre1.y_map5)
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

    def build_network(self):

        model = Sequential()
        vocabulary_size = len(self.tokenizer.word_index) + 1  # 学習データの語彙数+1z
        model.add(Embedding(input_dim=vocabulary_size,
                            output_dim=self.config.OUTPUT_DIM, mask_zero=True))
        model.add(LSTM(self.config.HIDDEN_UNITS,
                       return_sequences=False))
        model.add(Dense(25, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
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
            self.tokenizer = pickle.load(
                open(self.config.DATA_PATH + "/tokenizer.pkl", "rb"))
        else:
            self.init_data()
        print("finish init_data")
        self.network = self.build_network()
        self.network.summary()
        self.train_iterations()

    def train_iterations(self):
        print("start fit")
        print(np.array(self.X_train).shape)
        print(np.array(self.Y_train).shape)
        print(len(self.X_train[0]), self.X_train[0])
        print(self.network.predict(self.X_train))

        history = self.network.fit(self.X_train, self.Y_train, batch_size=self.config.BATCH_SIZE,
                                   epochs=self.config.EPOCH, validation_data=(
                                       self.X_test, self.Y_test),
                                   verbose=2)
        with open(self.config.OUTPUT_PATH + "/History.txt", "w") as f:
            f.write(history)
        self.network.save_weights(self.config.OUTPUT_PATH + "/param.h5")
        self.y_pred = self.network.predict_classes(self.X_test)
        print(confusion_matrix(self.Y_test, self.y_pred))
