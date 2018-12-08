import os
from datetime import datetime

import Stanford.Classifiers.utils as utils
import shared_utils


# A Random Forest classifier for the Stanford dataset
def main():
    data = utils.load_stanford_data_not_for_cnn(0.8, 0, 0.2)
    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = data

    num_features = train_set_x.shape[1]

    params = {
        'n_estimators': [200],
        # [5, 6, ..., 12]
        'max_depths': range(5, 13, 1),
        # [5, 10, ..., 50]
        'max_features': [int(0.01 * i * num_features) for i in range(5, 51, 5)],
        # [1, ..., 4]
        'min_samples_leafs': range(1, 5)
    }

    file_name = "RandomForestHyperParameters_" + str(int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())) + ".csv"
    results_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", file_name)

    shared_utils.random_forest_custom_grid_search(params, results_file_path, train_set_x, train_set_y, test_set_x, test_set_y)


if __name__ == '__main__':
    main()
