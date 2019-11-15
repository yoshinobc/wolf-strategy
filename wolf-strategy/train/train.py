from config import Config
from cnn import strategyCNN
import os
if __name__ == "__main__":
    config = Config()
    model = strategyCNN(config)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    # with open(config.OUTPUT_PATH + "/Config.txt", "w") as f:
    #    f.write(config.output_config())

    print("Training start")
    model.train()
