class Config(object):
    def __init__(self):
        self.EPOCH = 100
        self.LOG_PATH = "../AIWolf-ver0.5.6/log/*/*"
        self.OUTPUT_PATH = "result"
        self.BATCH_SIZE = 128
        self.TRAIN_PAR_TEST = 0.9
        self.MAX_LEN = 300
        self.HIDDEN_UNITS = 800
        self.OUTPUT_DIM = 200
        self.DATA_PATH = "result/data"
        self.INIT_DATA = True

    def output_config(self):
        text = "EPOCH: " + str(self.EPOCH) + "\n"
        text += "OUTPUT_PATH: " + self.OUTPUT_PATH + "\n"
        text += "LOG_PATH: " + self.LOG_PATH + "\n"
        text += "BATCH_SIZE: " + str(self.BATCH_SIZE) + "\n"
        text += "TRAIN_PAR_TEST: " + str(self.TRAIN_PAR_TEST) + "\n"
        text += "MAX_LEN" + str(self.MAX_LEN) + "\n"
        text += "HIDDEN_UNITS: " + str(self.HIDDEN_UNITS) + "\n"
        text += "OUTPUT_DIM: " + str(self.OUTPUT_DIM) + "\n"
        text += "DATA_PATH: " + self.DATA_PATH + "\n"
        text += "INIT_DATA: " + str(self.INIT_DATA) + "\n"
        return text
