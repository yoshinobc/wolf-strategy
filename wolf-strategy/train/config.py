class Config(object):
    def __init__(self):
        self.EPOCH = 10
        self.LOG_PATH = "../AIWolf-ver0.5.6/log/*/*"
        self.OUTPUT_PATH = "result-GCNN"
        self.BATCH_SIZE = 128
        self.TRAIN_PAR_TEST = 0.9
        self.LEARNING_RATE = 0.008
        self.DATA_PATH = "result-GCNN/data"
        self.INIT_DATA = False
        self.IS_FINISH_TRAIN = True
        self.TRAIN_FILE = "CNN"
        self.NETWORK = "GCNN"
        self.HIDDEN_UNITS = 200

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
