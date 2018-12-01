# Load the pickle file.
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from os.path import dirname, abspath
import utils

#parent_dir = dirname(abspath(__file__))

# An AdaBoost classifier for the UTD-MHAD dataset

def main():

    data = utils.load_utdmhad_cov_matrix_examples(0.8,0,0.2)
    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = data

    clf = AdaBoostClassifier(n_estimators=1000,learning_rate=0.001)

    clf.fit(train_set_X,train_set_y)
    print(clf.score(test_set_X, test_set_y))

if __name__ == '__main__':
    main()





