import pickle
import numpy as np
from os.path import dirname, abspath
import random
parent_dir = dirname(dirname(abspath(__file__)))
parent_parent_dir = dirname(parent_dir)

def load_MPII_data_categories_not_for_cnn(
    training_set_size,
    training_validation_set_size,
    testing_set_size
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
    MPII_dataset = pickle.load(open(parent_dir+ "/Data/MPII_dataset.p", "rb"))
    MPII_dataset_label_categories = pickle.load(open(parent_dir + "/Data/MPII_dataset_label_categories.p", "rb"))

    # [ [x,y] [x,y] .. ]

    # Get the number of images
    num_examples = len(MPII_dataset)

    #We need the x_data under the form [[32joins of image1], [32joins of image2], []...]
    X = np.reshape(MPII_dataset, (num_examples, 32))
    
    data = np.zeros((X.shape[0],X.shape[1],1))

    for i in range(num_examples):
        data[i] = []
    
    
    print("data: ", data)

    random.shuffle(MPII_dataset)

    num_train = int(num_examples * training_set_size)
    num_valid = int(num_examples * training_validation_set_size)
    num_test = int(num_examples * testing_set_size)

    bounds = [num_train - 1, num_train + num_valid - 1, num_train + num_valid + num_test]

    training_set = raw_dataset[:bounds[0], 1:]
    training_validation_set = raw_dataset[bounds[0]:bounds[1], 1:]
    testing_set = raw_dataset[bounds[1]:bounds[2], 1:]

    training_set_labels = raw_dataset[:bounds[0], 0]
    training_validation_labels = raw_dataset[bounds[0]:bounds[1], 0]
    testing_set_labels = raw_dataset[bounds[1]:bounds[2], 0]

