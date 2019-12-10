from config import Config
from cnn import strategyCNN
from lstm import strategyLSTM
import os

# agentのモデルを知っていればエージェントの役職推定がうまくいくか
# 1エージェントの性格の紹介
# 2input, outputで性格の推定
# 3役職推定がうまくいくか(性格予測の有無を使うか使わないか)
# 4
# 転移学習かラベルだけを役職推定に使う


# エージェント学習の転移学習で役職推定をする
# 同じ学習で大会エージェントの4つがクラスタリング．分類を行いない，どれぐらい区別ができるのか，性格のエージェントが区別できるのか

if __name__ == "__main__":
    config = Config()
    if config.TRAIN_FILE == "CNN":
        model = strategyCNN(config)
    else:
        model = strategyLSTM(config)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    # with open(config.OUTPUT_PATH + "/Config.txt", "w") as f:
    #    f.write(config.output_config())

    model.train()
