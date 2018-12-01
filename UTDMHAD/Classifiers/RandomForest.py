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
import random


# A Random Forest classifier for the UTD-MHAD dataset using the covariance matrix vector pre-processed data
def main():
    training_set_size = 0.8
    training_validation_set_size = 0
    testing_set_size = 0.2

    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = utils.load_utdmhad_cov_matrix_examples(
        training_set_size,
        training_validation_set_size,
        testing_set_size
    )
    num_dimensions = train_set_X.shape[1]

    # https://stackoverflow.com/questions/20463281/how-do-i-solve-overfitting-in-random-forest-of-python-sklearn
    # specify parameters and distributions to sample from
    # param_dist = {
    #     "max_depth": [3, None],
    #     "max_features": sp_randint(int(num_dimensions*0.3), int(num_dimensions*0.5)),
    #     "min_samples_split": sp_randint(2, 11),
    #     "min_samples_leaf": sp_randint(2, 10),
    #     "bootstrap": [True, False],
    #     "criterion": ["gini", "entropy"]
    # }
    #
    # # run randomized search
    # num_iterations = 1
    # model = RandomForestClassifier(n_estimators=100)
    # random_search = RandomizedSearchCV(model, param_distributions=param_dist,
    #                                    n_iter=num_iterations, cv=5)
    #
    # start = time()
    # random_search.fit(train_set_X, train_set_y)
    # print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #       " parameter settings." % ((time() - start), num_iterations))
    # shared_utils.report(random_search.cv_results_)






    model = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy', max_depth=7, max_features=720, min_samples_leaf=3)
    model.fit(train_set_X, train_set_y)
    print(model.score(train_set_X, train_set_y))
    print(model.score(test_set_X, test_set_y))















    # params_dict = {
    #     "max_depth": [random.choice([3, None]) for i in range(num_iterations)],
    #     "max_features": [random.randint(1, train_set_X.shape[1]) for i in range(num_iterations)],
    #     "min_samples_split": [random.randint(2, 10) for i in range(num_iterations)],
    #     "min_samples_leaf": [random.randint(2, 10) for i in range(num_iterations)],
    #     "bootstrap": [random.choice([True, False]) for i in range(num_iterations)],
    #     "criterion": [random.choice(["gini", "entropy"]) for i in range(num_iterations)]
    # }
    #
    # best_valid_accuracy = 0.0
    # for i in range(num_iterations):
    #     model = RandomForestClassifier(
    #         n_estimators=100,
    #         max_depth=params_dict["max_depth"][i],
    #         max_features=params_dict["max_features"][i],
    #         min_samples_split=params_dict["min_samples_split"][i],
    #         min_samples_leaf=params_dict["min_samples_leaf"][i],
    #         bootstrap=params_dict["bootstrap"][i],
    #         criterion=params_dict["criterion"][i]
    #     )
    #
    #     print("Iteration {}/{}".format(i, num_iterations))
    #
    #     model.fit(train_set_X, train_set_y)
    #     valid_accuracy = model.score(valid_set_X, valid_set_y)
    #     test_accuracy = model.score(test_set_X, test_set_y)
    #
    #     if valid_accuracy > best_valid_accuracy:
    #         best_valid_accuracy = valid_accuracy
    #         print("Test accuracy: {}. Validation accuracy: {} Hyper-parameters: {}".format(best_valid_accuracy, test_accuracy, model.get_params()))
    #
    # return
    #
    # # Try different max_depth (tree max depth) to improve results
    # model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    # model.fit(train_set_X, train_set_y)
    # acc = model.score(test_set_X, test_set_y)
    # print("Accuracy on the training set: %s \n" % acc)
    #
    # print("clf.feature_importances: %s \n " % model.feature_importances_)
    #
    # results = model.predict(test_set_X)
    #
    # results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "RandomForest_predictions.csv")
    # np.savetxt(results_path, results, delimiter=',', fmt='%s')
    #
    # print("Predictions have been saved as a csv file")


if __name__ == '__main__':
    main()
