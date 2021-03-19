import numpy as np
import matplotlib.pyplot as plt


def plot(count_dict, name):
    key_list = []
    value_list = []
    for key, value in count_dict.items():
        key_list.append(key)
        value_list.append(value)

    left = np.array(key_list)
    height = np.array(value_list)
    fix = plt.figure(figsize=(9.0, 6.0))
    plt.bar(left, height, tick_label=left, align="center")
    plt.savefig(name+".png")


all_dict = {'COMINGOUT': 763656, 'VOTE': 1267263,
            'ESTIMATE': 587384, 'DIVINED': 841553}

sample_dict = {'COMINGOUT': 145459, 'VOTE': 148943, 'DIVINED': 89416}

calm_dict = {'COMINGOUT': 175042, 'VOTE': 292379, 'DIVINED': 73531}

liar_dict = {'COMINGOUT': 175244, 'VOTE': 94285,
             'ESTIMATE': 399788, 'DIVINED': 557100}

repel_dict = {'COMINGOUT': 191812, 'VOTE': 289681, 'DIVINED': 73877}

follow_dict = {'COMINGOUT': 76099, 'VOTE': 441975,
               'ESTIMATE': 187596, 'DIVINED': 47629}

plot(all_dict, "all")
plot(sample_dict, "calups")
plot(calm_dict, "sonoda")
plot(liar_dict, "yskn")
plot(repel_dict, "cantar")
plot(follow_dict, "littlegirl")
