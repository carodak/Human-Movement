import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

import UTDMHAD.utils as utils


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

    # Try different max_depth (tree max depth) to improve results
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(train_set_X, train_set_y)
    acc = model.score(test_set_X, test_set_y)
    print("Accuracy on the training set: %s \n" % acc)

    return

    print("clf.feature_importances: %s \n " % model.feature_importances_)

    results = model.predict(test_set_X)

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "RandomForest_predictions.csv")
    np.savetxt(results_path, results, delimiter=',', fmt='%s')

    print("Predictions have been saved as a csv file")


if __name__ == '__main__':
    main()
