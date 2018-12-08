import pickle
import numpy as np
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
import random

def load_MPII_data_not_for_cnn(
    training_set_size,
    training_validation_set_size,
    testing_set_size,
    act_cat,
    use_dist
):

    """
        :param training_set_size: The % of the dataset we want as the training set [0,1]
        :param training_validation_set_size: The % of the dataset we want as the training validation set [0,1]
        :param testing_set_size: The % of the dataset we want as the testing set [0,1]
        :return: training_set, training_set_labels,
                 training_validation_set, training_validation_labels,
                 testing_set, testing_set_labels
    """

    assert (0 <= training_set_size <= 1)
    assert (0 <= training_validation_set_size <= 1)
    assert (0 <= testing_set_size <= 1)
    assert (0 <= training_set_size + training_validation_set_size + testing_set_size <= 1)

    # Load the data
    with open( parent_dir+"/Data/MPII_dataset.p", "rb" ) as f:
        dataset = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_activities.p", "rb" ) as f:
        activities = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_images_names.p", "rb" ) as f:
        images_names = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_label_categories.p", "rb" ) as f:
        categories = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_label.p", "rb" ) as f:
        label = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_euclidean_distance.p", "rb" ) as f:
        distance_euc = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_minkowski_p1.p", "rb" ) as f:
        distance_min_p1 = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_minkowski_p3.p", "rb" ) as f:
        distance_min_p3 = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_cosine.p", "rb" ) as f:
        distance_cos = pickle.load(f)
    


    if act_cat == "cat":
        Y = categories
    else:
        Y = activities

    if use_dist == 0:
        num_examples = len(dataset)
        X = np.reshape(dataset, (num_examples, 32))

    if use_dist == 1:
        num_examples = distance_euc.shape[0]
        X = np.reshape(distance_euc, (num_examples, 16*16))

    if use_dist == 2:
        num_examples = distance_min_p1.shape[0]
        X = np.reshape(distance_min_p1, (num_examples, 16*16))

    if use_dist == 3:
        num_examples = distance_min_p3.shape[0]
        X = np.reshape(distance_min_p3, (num_examples, 16*16))

    if use_dist == 4:
        num_examples = distance_cos.shape[0]
        X = np.reshape(distance_cos, (num_examples, 16*16))

    raw_dataset = np.array([np.append(X[i],Y[i])  for i in range(num_examples)])

#print(raw_dataset[0])

    random.Random(4).shuffle(raw_dataset)

    num_train = int(num_examples * training_set_size)
    num_valid = int(num_examples * training_validation_set_size)
    num_test = int(num_examples * testing_set_size)

    bounds = [num_train - 1, num_train + num_valid - 1, num_train + num_valid + num_test]

    training_set = raw_dataset[:bounds[0], :-1]
    training_validation_set = raw_dataset[bounds[0]:bounds[1], :-1]
    testing_set = raw_dataset[bounds[1]:bounds[2], :-1]

    training_set_labels = raw_dataset[:bounds[0], -1]
    training_validation_labels = raw_dataset[bounds[0]:bounds[1], -1]
    testing_set_labels = raw_dataset[bounds[1]:bounds[2], -1]

#print(testing_set_labels)

    return training_set, training_set_labels,\
        training_validation_set, training_validation_labels,\
            testing_set, testing_set_labels
