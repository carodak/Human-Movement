import numpy as np
import random


def load_utd_mhad():
    UTD_dataset = np.loadtxt("/Data/UTD_MHAD_Labeled_Descriptors.csv", delimiter=",", skiprows=0)

    # Commenter pour avoir des resultats non-deterministes
    random.seed(5)

    ##########################  Make validation set with 1/3 of training set ########################

    n_train = int(UTD_dataset.shape[1] * 0.66)

    # Determiner au hasard des indices pour les exemples d'entrainement et de test
    inds = list(range(UTD_dataset.shape[1]))

    random.shuffle(inds)
    knn_train_for_val_inds = inds[:n_train]
    knn_valid_inds = inds[n_train:]

    TRAIN_labled_set_for_valid = UTD_dataset[:, knn_train_for_val_inds].T
    test_labled_set_for_valid = UTD_dataset[:, knn_valid_inds].T

    train_labels = TRAIN_labled_set_for_valid[:, 0]

    train_descriptors = TRAIN_labled_set_for_valid[:, 1:]

    test_labels = test_labled_set_for_valid[:, 0]

    test_descriptors = test_labled_set_for_valid[:, 1:]

    test_predictions = np.zeros(test_labled_set_for_valid.shape[0])