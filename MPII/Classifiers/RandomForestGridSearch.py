import os
import sys
from datetime import datetime

import MPII.Classifiers.utils as utils
import shared_utils


# A Random Forest classifier for the MPII dataset
def main():
    choice = input("Activities or Categories? (act/cat)")
    choice = choice.lower()

    if choice not in ['act', 'cat']:
        print("Only the choices act or cat are supported")
        sys.exit(1)

    data = utils.load_MPII_data_not_for_cnn(0.8, 0, 0.2, choice, 0)
    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = data

    num_features = train_set_x.shape[1]

    params = {
        'n_estimators': [200],
        # [5, 6, ..., 12]
        'max_depths': range(5, 12, 1),
        # [5, 10, ..., 50]
        'max_features': [int(0.01 * i * num_features) for i in range(5, 31, 5)],
        'min_samples_leafs': [1, 2, 3]
    }

    file_name = '_'.join(["RandomForestHyperParameters", choice, str(int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()))])
    results_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", file_name)

    shared_utils.random_forest_custom_grid_search(params, results_file_path, train_set_x, train_set_y, test_set_x, test_set_y)
    #shared_utils.save_plots_for_random_forest_grid_search_results(results_file_path)


if __name__ == '__main__':
    main()
