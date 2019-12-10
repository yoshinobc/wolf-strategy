import numpy as np
import matplotlib.pyplot as plt
def plot(count_dict,name):
    key_list = []
    value_list = []
    for key, value in count_dict.items():
        key_list.append(key)
        value_list.append(value)

    left = np.array(key_list)
    height = np.array(value_list)
    fix = plt.figure(figsize=(9.0, 6.0))
    plt.bar(left, height,tick_label=left, align="center")
    plt.savefig(name+".png")

all_dict = {'COMINGOUT': 634072, 'INQUIRE': 179354, 'BECAUSE': 722438, 'REQUEST': 2230104, 'DISAGREE': 802439, 'DAY': 252538, 'ESTIMATE': 751327, 'VOTE': 3519579, 'DIVINED': 268272, 'AGREE': 1097805}

sample_dict = {'COMINGOUT': 81929, 'BECAUSE': 112966, 'REQUEST': 321456, 'DAY': 252538, 'ESTIMATE': 86769, 'VOTE': 284909, 'DIVINED': 165636}

calm_dict = {'COMINGOUT': 293401, 'BECAUSE': 166834, 'REQUEST': 309830, 'DISAGREE': 517951, 'ESTIMATE': 333668, 'VOTE': 648726, 'DIVINED': 19449, 'AGREE': 270526}

liar_dict = {'COMINGOUT': 122591, 'INQUIRE': 144087, 'BECAUSE': 277193, 'DISAGREE': 284488, 'VOTE': 1022899, 'DIVINED': 29723, 'AGREE': 255747}

repel_dict = {'COMINGOUT': 123386, 'BECAUSE': 165445, 'REQUEST': 118580, 'ESTIMATE': 330890, 'VOTE': 615030, 'DIVINED': 53464, 'AGREE': 366072}

follow_dict = {'COMINGOUT': 12765, 'INQUIRE': 35267, 'VOTE': 948015, 'REQUEST': 1480238, 'AGREE': 205460}
plot(all_dict,"all")
plot(sample_dict,"sample")
plot(calm_dict,"calm")
plot(liar_dict,"liar")
plot(repel_dict,"repel")
plot(follow_dict,"follow")
