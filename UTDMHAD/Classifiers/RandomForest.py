import numpy as np  # linear algebra
import os

from time import time
from scipy.stats import randint as sp_randint
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import UTDMHAD.Classifiers.utils as utils
import shared_utils
from functools import reduce


# https://stackoverflow.com/questions/20463281/how-do-i-solve-overfitting-in-random-forest-of-python-sklearn
# A Random Forest classifier for the UTD-MHAD dataset using the covariance matrix vector pre-processed data
def main():
    training_set_size = 0.8
    training_validation_set_size = 0
    testing_set_size = 0.2

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = utils.load_utdmhad_cov_matrix_examples(
        training_set_size,
        training_validation_set_size,
        testing_set_size
    )

    # basic_unoptimised_run_example(train_set_x, train_set_y, test_set_x, test_set_y)
    # randomised_search(model)
    custom_grid_search(train_set_x, train_set_y, test_set_x, test_set_y)


def basic_unoptimised_run_example(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y
):
    # Try different max_depth (tree max depth) to improve results
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(train_set_x, train_set_y)
    acc = model.score(test_set_x, test_set_y)

    print("Accuracy on the training set: %s \n" % acc)
    print("clf.feature_importances: %s \n " % model.feature_importances_)


def custom_grid_search(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y
):
    num_examples = train_set_x.shape[0]

    # params = {
    #     'n_estimators': range(20, 200, 20),
    #     'max_depths': range(5, 20),
    #     # 30-50% of the data set
    #     'max_features': [int(0.01*i*num_examples) for i in range(30, 50, 5)],
    #     'min_samples_leafs': range(1, 20)
    # }

    params = {
        'n_estimators': range(100, 200, 50),
        'max_depths': range(5, 20, 1),
        # 30-50% of the data set
        'max_features': [int(0.01 * i * num_examples) for i in range(30, 50, 5)],
        'min_samples_leafs': range(1, 10)
    }

    average_runs_count = 5
    iteration = 1
    total_iterations = reduce(lambda x, y: x*y, [len(lst) for lst in params.values()])
    file_name = "RandomForestHyperParameters_" + str(int((datetime.utcnow()-datetime(1970,1,1)).total_seconds())) + ".csv"
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", file_name)
    file_results = open(file_path, "w+")
    file_results.write(','.join(["train_accuracy", "test_accuracy", "n_estimator", "max_depth", "max_feature", "min_samples_leaf", "\n"]))
    for n_estimator in params['n_estimators']:
        for max_depth in params['max_depths']:
            for max_feature in params['max_features']:
                for min_samples_leaf in params['min_samples_leafs']:
                    print("Iteration {}/{}".format(iteration, total_iterations))

                    train_accuracy = 0
                    test_accuracy = 0

                    for i in range(average_runs_count):
                        model = RandomForestClassifier(
                            n_estimators=n_estimator,
                            max_depth=max_depth,
                            max_features=max_feature,
                            min_samples_leaf=min_samples_leaf,
                            bootstrap=True,
                            criterion='entropy'
                        )

                        model.fit(train_set_x, train_set_y)
                        train_accuracy += model.score(train_set_x, train_set_y)
                        test_accuracy += model.score(test_set_x, test_set_y)

                    train_accuracy /= (1.0*average_runs_count)
                    test_accuracy /= (1.0*average_runs_count)

                    file_results.write(','.join(str(x) for x in [train_accuracy, test_accuracy, n_estimator, max_depth, max_feature, min_samples_leaf]))
                    file_results.write('\n')
                    iteration += 1

    file_results.close()


def randomised_search(
    model,
    train_set_x,
    train_set_y
):
    num_dimensions = train_set_x.shape[1]

    # specify parameters and distributions to sample from
    param_dist = {
        "max_depth": [3, None],
        "max_features": sp_randint(int(num_dimensions*0.3), int(num_dimensions*0.5)),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(2, 10),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }

    # run randomized search
    num_iterations = 1
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=num_iterations, cv=5)

    start = time()
    random_search.fit(train_set_x, train_set_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), num_iterations))
    shared_utils.report(random_search.cv_results_)


def predict_and_save_results(
    model
):
    results = model.predict(test_set_X)

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "RandomForest_predictions.csv")
    np.savetxt(results_path, results, delimiter=',', fmt='%s')

    print("Predictions have been saved as a csv file")


if __name__ == '__main__':
    main()
