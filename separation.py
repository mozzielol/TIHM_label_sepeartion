import numpy as np
from keras.utils import to_categorical

"""
Data Pre processing
"""
"""
Assumptions:
1. Take the transitions as vectors
"""
trainsition_matrix = np.concatenate([
    np.arange(8).reshape(1, 1, 1, 8) * 8,
    np.arange(8).reshape(1, 1, 1, 8)
], axis=2)


def separate_each_slots(data, step, trans=False):
    data = np.transpose(data, (0, 2, 1))
    data[np.where(data != 0.)] = 1
    sliced_data = {}
    for t in np.arange(0, 24 * step, 2):
        sliced_data[t] = []
        sliced_data[t].append(data[:, t::24 * step])
        sliced_data[t].append(data[:, (t + 1)::24 * step])
        try:
            sliced_data[t] = np.squeeze(np.concatenate(sliced_data[t]), axis=1)
            sliced_data[t] = sliced_data[t][~np.all(sliced_data[t] == 0, axis=-1)]
            remains = sliced_data[t].shape[0]
            if remains % 2 > 0:
                sliced_data[t] = np.delete(sliced_data[t], 0, axis=0)
                remains = sliced_data[t].shape[0]
            sliced_data[t] = np.sum(np.sum(sliced_data[t].reshape(remains // 2, 2, 8) * trainsition_matrix, axis=-1),
                          axis=-1)  # build the matrix by counting, shape = (-1, )
            sliced_data[t] = sliced_data[t].reshape(-1, 1)

        except ValueError:
            sliced_data.pop(t)
        if trans:
            return to_categorical(sliced_data[t], num_classes=int(np.max(sliced_data[t])) + 1)

    return sliced_data


def external_features(data, trans=True):
    # define the activity patterns/level
    data = np.transpose(data, (0, 2, 1))  # transpose matrix, data = (-1, 1440, 8)
    data[np.where(data != 0.)] = 1  # all the sensors are triggered by one person in this slot
    data = data[~np.all(data == 0, axis=-1)]  # remove all the rows if no sensors are triggered
    remains = data.shape[0]
    if remains % 2 > 0:
        data = np.delete(data, 0, axis=0)
        remains = data.shape[0]
    data = np.sum(np.sum(data.reshape(remains // 2, 2, 8) * trainsition_matrix, axis=-1),
                  axis=-1)  # build the matrix by counting, shape = (-1, )
    data = data.reshape(-1, 1)
    if trans:
        return to_categorical(data, num_classes=int(np.max(data)) + 1)
    return data


def external_features_no_deletion(data, trans=True):
    # define the activity patterns/level
    data = np.transpose(data, (0, 2, 1))  # transpose matrix, data = (-1, 1440, 8)
    data[np.where(data != 0.)] = 1  # all the sensors are triggered by one person in this slot
    remains = data.shape[0]
    if remains % 2 > 0:
        data = np.delete(data, 0, axis=0)
        remains = data.shape[0]
    data = np.sum(np.sum(data.reshape(remains, data.shape[1]//2, 2, 8) * trainsition_matrix, axis=-1),
                  axis=-1)  # build the matrix by counting, shape = (-1, )
    data = data.reshape(remains, -1)
    if trans:
        return to_categorical(data, num_classes=int(np.max(data)) + 1)
    return data


"""
Algorithm
"""
# To separate the labels, we can do Three things
"""
1. Assume there is only one subject move around within each hour at each day. 
2. If we can define the activity level, we can assume the patterns are similar in each day. And three types of movement, patient, carers and both.
3. Assume the patterns are similar in each day. And TWO types of movement, patient or carers.
4. we can cluster the pattern of each hour within each day.
"""

from sklearn import mixture


# Assumption 3
def GMM_clustering(data, n_components=2):
    try:
        model = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(data)
    except ValueError:
        return None
    return model
