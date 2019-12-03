import numpy as np
import matplotlib.pyplot as plt
def plot(count_dict):
    key_list = []
    value_list = []
    for key, value in count_dict.items():
        key_list.append(key)
        value_list.append(value)

    left = np.array(key_list)
    height = np.array(value_list)
    fix = plt.figure(figsize=(9.0, 6.0))
    plt.bar(left, height,tick_label=left, align="center")
    plt.show()

all_dict = {'BECAUSE': 892618, 'COMINGOUT': 1041994, 'ESTIMATE': 1577232, 'INQUIRE': 80289, 'DIVINED': 441716, 'VOTE': 4431009, 'REQUEST': 3266716, 'DAY': 415393, 'DISAGREE': 915777}

sample_dict = {'BECAUSE': 203222, 'COMINGOUT': 150694, 'ESTIMATE': 198440, 'DIVINED': 266788, 'VOTE': 473527, 'REQUEST': 530173, 'DAY': 415393}

calm_dict = {'BECAUSE': 280360, 'COMINGOUT': 479548, 'ESTIMATE': 560720, 'DIVINED': 35195, 'VOTE': 1092545, 'REQUEST': 511923, 'DISAGREE': 915777}

liar_dict = {'BECAUSE': 205750, 'COMINGOUT': 198725, 'ESTIMATE': 411500, 'DIVINED': 70782, 'VOTE': 773395, 'REQUEST': 155949}

repel_dict = {'BECAUSE': 203286, 'COMINGOUT': 197452, 'ESTIMATE': 406572, 'DIVINED': 68951, 'VOTE': 763859, 'REQUEST': 153803}

follow_dict = {'COMINGOUT': 15575, 'VOTE': 1327683, 'REQUEST': 1914868, 'INQUIRE': 80289}

plot(all_dict)
plot(sample_dict)
plot(calm_dict)
plot(liar_dict)
plot(repel_dict)
plot(follow_dict)
