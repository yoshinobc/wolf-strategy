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


class strategyLSTM(object):
    def __init__(self, config):
        self.config = config

    def init_data(self):
        file_names = glob(self.config.LOG_PATH)
        texts = []
        texts_y = []
        for name in tqdm(file_names):
            f = open(name, mode="r")
            c = f.readline()
            contents_y = np.zeros((5, 5))
            text = []
            text_ = []
            while c:
                c = f.readline().rstrip(os.linesep)
                contents = c.split(",")
                contents_tmp = []
                if len(contents) <= 1:
                    continue
                if contents[1] == "status":
                    if "Sample" in contents[5]:
                        agent = int(contents[2]) - 1
                        contents_y[agent][0] = 1

                    elif "CALM" in contents[5]:
                        agent = int(contents[2]) - 1
                        contents_y[agent][1] = 1

                    elif "Liar" in contents[5]:
                        agent = int(contents[2]) - 1
                        contents_y[agent][2] = 1

                    elif "REPEL" in contents[5]:
                        agent = int(contents[2]) - 1
                        contents_y[agent][3] = 1

                    elif "Follow" in contents[5]:
                        agent = int(contents[2]) - 1
                        contents_y[agent][4] = 1
                    continue
                elif contents[1] == "talk":
                    day = contents[0]
                    tp = contents[1]
                    id_ = contents[2]
                    agent = contents[4]
                    sentence = contents[5].split(" ")
                    contents_tmp = [day, tp, id_, agent]
                    # print(sentence)
                    if sentence[0] == "AGREE" or sentence[0] == "DISAGREE":
                        d = int(sentence[2][3:])
                        i = int(sentence[3][3:6])
                        for te in text:
                            if int(te[0]) == d and int(te[2]) == i and te[1] == "talk":
                                # print(contents[5])
                                contents_tmp.extend([sentence[0]])
                                contents_tmp.extend(te[4:])
                    else:
                        contents_tmp.extend(sentence)
                    # .extend(sentence)
                elif contents[1] == "divine" and contents[2] == "1":
                    day = contents[0]
                    tp = contents[1]
                    agent = contents[3]
                    sentence = contents[4]
                    contents_tmp = [day, tp, agent, sentence]
                elif contents[1] == "vote" or contents[1] == "execute" or contents[1] == "attack":
                    day = contents[0]
                    tp = contents[1]
                    agent = contents[2]
                    sentence = contents[3]
                    contents_tmp = [day, tp, agent, sentence]
                else:
                    continue
                text.append(contents_tmp)
                contents_tmp = " ".join(contents_tmp)
                text_.append(contents_tmp)
                # text_ += contents_tmp + "\n"
            texts.append(text_)
            flaten_contents_y = np.array(contents_y).flatten()
            texts_y.append(flaten_contents_y)
            f.close()
        X_train, X_test = texts[:int(len(
            texts)*self.config.TRAIN_PAR_TEST)], texts[int(len(texts)*self.config.TRAIN_PAR_TEST) + 1:]
        self.Y_train, self.Y_test = texts_y[:int(len(
            texts)*self.config.TRAIN_PAR_TEST)], texts_y[int(len(texts)*self.config.TRAIN_PAR_TEST) + 1:]
        print("init tokenizer")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X_train)
        print("fit tokenizer")
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)
        self.X_train = pad_sequences(X_train, maxlen=self.config.MAX_LEN)
        self.X_test = pad_sequences(X_test, maxlen=self.config.MAX_LEN)
        os.makedirs(self.config.OUTPUT_PATH+"/data", exist_ok=True)
        pickle.dump(self.X_train, open(self.config.OUTPUT_PATH +
                                       "/data/X_train.pkl", mode="wb"))
        pickle.dump(self.X_test, open(self.config.OUTPUT_PATH +
                                      "/data/X_test.pkl", mode="wb"))
        pickle.dump(self.Y_train, open(self.config.OUTPUT_PATH +
                                       "/data/y_train.pkl", mode="wb"))
        pickle.dump(self.Y_test, open(self.config.OUTPUT_PATH +
                                      "/data/y_test.pkl", mode="wb"))
        pickle.dump(self.tokenizer, open(
            self.config.OUTPUT_PATH + "/data/tokenizer.pkl", mode="wb"))

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
        """

        main_input = Input(shape=(np.array(self.X_train).shape[1],))
        embedding = Embedding(
            input_dim=self.config.MAX_LEN,
            output_dim=self.config.HIDDEN_UNITS,
            trainable=False,
            mask_zero=True
        )(main_input)
        lstm = LSTM(32)(embedding)
        main_output = Dense(25)(lstm)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss="mse", optimizer="adam")
        return model
        """

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
        # print(self.X_train[0])
        # print(self.Y_train[0])
        print(np.array(self.Y_train).shape)
        print(len(self.X_train[0]), self.X_train[0])
        print(self.network.predict(self.X_train))
        # 出力の形式があってない？
        # print(self.X_train[0])
        history = self.network.fit(self.X_train, self.Y_train, batch_size=self.config.BATCH_SIZE,
                                   epochs=self.config.EPOCH, validation_data=(
                                       self.X_test, self.Y_test),
                                   verbose=2)
        with open(self.config.OUTPUT_PATH + "/History.txt", "w") as f:
            f.write(history)
        self.network.save_weights(self.config.OUTPUT_PATH + "/param.h5")
        self.y_pred = self.network.predict_classes(self.X_test)
        print(confusion_matrix(self.Y_test, self.y_pred))
