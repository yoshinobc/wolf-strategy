from config import Config
from lstm import strategyLSTM

if __name__ == "__main__":
    config = Config()
    model = strategyLSTM(config)
    with open(config.OUTPUT_PATH + "/Config.txt", "w") as f:
        f.write(config.output_config())

    print("Training start")
    model.train()
    """
    if config.RESUME_TRAIN:
        discriminator_path = "./result/181104_1905/weights/ramen_cam/discriminator245000.hdf5"
        generator_path = "./result/181104_1905/weights/ramen_cam/generator245000.hdf5"
        print("Training start at {} iterations".format(config.COUNTER))
        model.resume_train(discriminator_path, generator_path, config.COUNTER)
    else:
        print("Training start")
        model.train()
    """
