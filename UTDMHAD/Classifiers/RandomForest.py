import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import UTDMHAD.Classifiers.utils as utils
import shared_utils


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
    num_features = train_set_x.shape[1]

    choice = shared_utils.grid_or_spec()

    if choice == 'grid':
        params = {
            'n_estimators': [200],
            # [5, 6, ..., 12]
            'max_depths': range(5, 13, 1),
            # [5, 10, ..., 50]
            'max_features': [int(0.01 * i * num_features) for i in range(5, 51, 5)],
            'min_samples_leafs': [1, 2, 3]
        }

        file_name = "RandomForestHyperParameters_" + str(int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()))
        results_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", file_name)

        shared_utils.random_forest_custom_grid_search(params, results_file_path, 5, train_set_x, train_set_y, test_set_x, test_set_y)
        shared_utils.save_plots_for_random_forest_grid_search_results(results_file_path)

    elif choice == 'spec':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            max_features=137,
            min_samples_leaf=1,
            bootstrap=True,
            criterion='entropy'
        )

        model.fit(train_set_x, train_set_y)
        print("Training accuracy: {}".format(model.score(train_set_x, train_set_y)))
        print("Test accuracy: {}".format(model.score(test_set_x, test_set_y)))

        conf_matrix = shared_utils.confusion_matrix(test_set_y, model.predict(test_set_x))
        # Confusion matrix as a heatmap:
        plt.imshow(conf_matrix, cmap='hot', interpolation='nearest')
        plt.show()


if __name__ == '__main__':
    main()
