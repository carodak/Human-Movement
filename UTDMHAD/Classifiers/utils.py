import numpy as np
import random
import os.path


def load_utdmhad_cov_matrix_examples(
    training_set_size,
    training_validation_set_size,
    testing_set_size
):
    """
    Load the covariance matrix vector method pre-processed UTD-MHAD dataset

    :param training_set_size: The % of the dataset we want as the training set [0,1]
    :param training_validation_set_size: The % of the dataset we want as the training validation set [0,1]
    :param testing_set_size: The % of the dataset we want as the testing set [0,1]
    :return: training_set, training_set_labels,
             training_validation_set, training_validation_labels,
             testing_set, testing_set_labels
    """
    assert(0 <= training_set_size <= 1)
    assert(0 <= training_validation_set_size <= 1)
    assert(0 <= testing_set_size <= 1)
    assert(0 <= training_set_size + training_validation_set_size + testing_set_size <= 1)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "UTD_MHAD_Labeled_Descriptors.csv")
    raw_dataset = np.loadtxt(data_path, delimiter=",", skiprows=0)
    raw_dataset = raw_dataset.T
    num_examples = raw_dataset.shape[0]

    random.shuffle(raw_dataset)

    num_train = int(num_examples * training_set_size)
    num_valid = int(num_examples * training_validation_set_size)
    num_test = int(num_examples * testing_set_size)

    bounds = [num_train-1, num_train+num_valid-1, num_train+num_valid+num_test]

    training_set = raw_dataset[:bounds[0], 1:]
    training_validation_set = raw_dataset[bounds[0]:bounds[1], 1:]
    testing_set = raw_dataset[bounds[1]:bounds[2], 1:]

    training_set_labels = raw_dataset[:bounds[0], 0]
    training_validation_labels = raw_dataset[bounds[0]:bounds[1], 0]
    testing_set_labels = raw_dataset[bounds[1]:bounds[2], 0]

    return training_set, training_set_labels,\
           training_validation_set, training_validation_labels,\
           testing_set, testing_set_labels
