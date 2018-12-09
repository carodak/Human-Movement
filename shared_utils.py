from functools import reduce
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def random_forest_custom_grid_search(
    params,
    results_file_path,
    average_runs_count,
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
):
    """
    Do a grid search to find the best hyper-parameters for a RandomForest algorithm on the data sets provided

    :param params: A dictionary of the sklearn.RandomForestClassifier parameters
        n_estimators, max_depths, max_features and min_samples_leafs to test
    :param results_file_path: The absolute path to the file the results from the grid search will be stored
    :param average_runs_count: How many times should each combination be tried?
    :param train_set_x: The inputs for the training set
    :param train_set_y: The corresponding outputs (labels) for the training set
    :param test_set_x: The inputs for the test set
    :param test_set_y: The corresponding outputs (labels) for the test set
    """
    iteration = 1
    total_iterations = reduce(lambda x, y: x*y, [len(lst) for lst in params.values()])
    file_results = open(results_file_path+".csv", "w+")
    file_results.write(','.join(["overfitting", "train_accuracy", "test_accuracy", "n_estimators", "max_depth", "max_features", "min_samples_leaf", "\n"]))
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

                        # Re-shuffle data sets for better averaging accuracy
                        train_set_x, train_set_y = shuffle_data_sets(train_set_x, train_set_y)
                        test_set_x, test_set_y = shuffle_data_sets(test_set_x, test_set_y)

                    train_accuracy /= (1.0*average_runs_count)
                    test_accuracy /= (1.0*average_runs_count)

                    file_results.write(','.join(str(x) for x in [train_accuracy-test_accuracy, train_accuracy, test_accuracy, n_estimator, max_depth, max_feature, min_samples_leaf]))
                    file_results.write('\n')
                    iteration += 1

    file_results.close()


def shuffle_data_sets(
    set_X,
    set_y
):
    """
    Shuffle data set with inputs X and outputs (labels) y preserving the mapping of input x1 to output y1

    :param set_X: The data set inputs
    :param set_y: The data set outputs
    :return: set_X, set_y shuffled in the same way
    """
    shuffled_train_set = list(zip(set_X, set_y))
    random.shuffle(shuffled_train_set)
    return zip(*shuffled_train_set)


def save_plots_for_random_forest_grid_search_results(
        csv_file_name
):
    """
    Save the plots for the results in the random_forest_grid_search_csv_results_file in the same folder as that file

    :param csv_file_name: The RandomForest hyper-parameter grid search results csv file
    """
    hyper_params_results = np.genfromtxt(csv_file_name+'.csv', dtype=float, delimiter=',', names=True)

    # [x, y, z] coordinate indexes taken from hyper_params_results for the 3D graphs
    combinations = [[5, 4, 2], [6, 4, 2], [6, 5, 2]]

    for combination in combinations:
        labels = np.array([hyper_params_results.dtype.names[combination[i]] for i in range(0, len(combinations))])
        hyper_params_results.sort(order=[labels[0], labels[1]])

        coords = np.zeros((len(combination), hyper_params_results.shape[0]))
        for j in range(0, len(combinations)):
            coords[j] = np.array([hyper_params_results[i][combination[j]] for i in range(0, len(hyper_params_results))])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.update({'xlabel': labels[0], 'ylabel': labels[1], 'zlabel': labels[2]})
        surf = ax.plot_trisurf(coords[0], coords[1], coords[2], cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig(csv_file_name+'-'.join(labels)+'.png', dpi=100)


def confusion_matrix(
    true_labels,
    predicted_labels
):
    """
    Constructs a confusion matrix based on the true and predicted labels given

    :param true_labels:
    :param predicted_labels:
    :return:
    """
    n_classes = int(max(true_labels + 1))
    matrix = np.zeros((n_classes, n_classes))

    for (test, pred) in zip(true_labels, predicted_labels):
        matrix[int(test), int(pred)] += 1

    return matrix
