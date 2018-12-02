import pickle
import numpy as np
import os
import random

def load_stanford_data_not_for_cnn(
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
    path = os.path.join('..','Data')
    with open(os.path.join(path, 'stanford_joints_pose.p'),'rb') as f:
        dataset = pickle.load(f)
    with open(os.path.join(path, 'stanford_labels.p'),'rb') as f:
        label = pickle.load(f)


    dataset = np.array([x.tolist() for x in dataset])
    dataset = dataset[:,:,:-1]
    num_examples = dataset.shape[0]
    dataset = np.reshape(dataset, (num_examples, 36))

    raw_dataset = np.array([np.append(dataset[i],label[i])  for i in range(num_examples)])
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
