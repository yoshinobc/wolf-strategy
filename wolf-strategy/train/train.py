from config import Config
from cnn import strategyCNN
from lstm import strategyLSTM
import os
if __name__ == "__main__":
    config = Config()
    if config.TRAIN_FILE == "CNN":
        model = strategyCNN(config)
    else:
        model = strategyLSTM(config)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    # with open(config.OUTPUT_PATH + "/Config.txt", "w") as f:
    #    f.write(config.output_config())

    print("Training start")
    model.train()
