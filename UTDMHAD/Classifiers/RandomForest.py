import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    clf.fit(train_set_X, train_set_y)
    acc = clf.score(train_set_X, train_set_y)
    print("Accuracy on the training set: %s \n" % acc)

    print("clf.feature_importances: %s \n " % clf.feature_importances_)

    Xtest_ = images_train[:, -1]
    nt = len(Xtest_)
    nt2 = len(Xtest_[0])
    Xtest = np.zeros([nt, nt2])
    for i in range(nt):
        Xtest[i] = Xtest_[i].tolist()

    results = clf.predict(Xtest)

    print("Results (predictions on the test set): %s \n" % results)

    np.savetxt(parent_dir + '/results/test_random_forest.csv', [p for p in results], delimiter=' ', fmt='%s')

    print("Predictions as been saved as a csv file")


if __name__ == '__main__':
    main()
