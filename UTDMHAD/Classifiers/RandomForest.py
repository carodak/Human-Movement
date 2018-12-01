import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

import UTDMHAD.utils as utils
import shared_utils


# A Random Forest classifier for the UTD-MHAD dataset using the covariance matrix vector pre-processed data
def main():
    training_set_size = 0.01
    training_validation_set_size = 0
    testing_set_size = 0

    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = utils.load_utdmhad_cov_matrix_examples(
        training_set_size,
        training_validation_set_size,
        testing_set_size
    )

    # build a classifier
    model = RandomForestClassifier(n_estimators=100)

    # specify parameters and distributions to sample from
    param_dist = {
        "max_depth": [3, None],
        "max_features": sp_randint(1, train_set_X.shape[1]),
        "min_samples_leaf": sp_randint(1, 10),
        "min_samples_split": sp_randint(2, 10),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search, cv=5)

    start = time()
    random_search.fit(train_set_X, train_set_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    shared_utils.report(random_search.cv_results_)

    return

    # Try different max_depth (tree max depth) to improve results
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(train_set_X, train_set_y)
    acc = model.score(test_set_X, test_set_y)
    print("Accuracy on the training set: %s \n" % acc)

    print("clf.feature_importances: %s \n " % model.feature_importances_)

    results = model.predict(test_set_X)

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "RandomForest_predictions.csv")
    np.savetxt(results_path, results, delimiter=',', fmt='%s')

    print("Predictions have been saved as a csv file")


if __name__ == '__main__':
    main()
