from data_utils import load_data
from separation import external_features_no_deletion, GMM_clustering, separate_each_slots
from visualisation import plot_1d
import numpy as np

# step = 1
# interval = 60 / step
# data = load_data('DRI_incident', return_one=True)
# data = separate_each_slots(data, 60)
#
# for key in data.keys():
#     model = GMM_clustering(data[key], 3)
#     if model is not None:
#         plot_1d(model, 'sliced60min/' + str(key), '%d clock %d min to %d min' % (key // 4, key / 2 % 2 * interval * 2, key / 2 % 2 * interval * 2 + interval * 2))


# data = load_data('DRI_env', return_one=True)
# daliy_data = external_features_no_deletion(data, False)
#
# model = GMM_clustering(daliy_data, 3)
# plot_1d(model, 'one_day', 'Daily Patterns')

""" 8 hours daily"""
data = load_data('DRI_env', return_one=True)
for t in range(3):
    daily_data = data[t::3]
    for _ in range(2):
        daliy_data = external_features_no_deletion(daily_data, False)
    model = GMM_clustering(daliy_data, 3)
    plot_1d(model, '8hours/%d' % t, 'Daily Patterns %d to %d clock' % (t * 8, (t + 1) * 8))

# Configurations: Database, one hot or not, number of cluster, figure name
# configurations = [
#     ['DRI_env',  True, 3, '3trans1'],  # 1 min span, transition features
#     ['TIHM_env', True, 3, '3trans15'],  # 15 min span, transition features
#     ['DRI_env', False, 3, '3act1'],  # 1 min span, measure activity
#     ['DRI_env', False, 3, '3act15'],  # 15 min span, measure activity
# ]
#
# for conf in configurations:
#     data = load_data(conf[0], return_one=True)
#     data = external_features(data, conf[1])
#
#     model = GMM_clustering(data, conf[2])
#     plot_1d(model, conf[3])
