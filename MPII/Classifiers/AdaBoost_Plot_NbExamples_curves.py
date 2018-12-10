
# Load the pickle file.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import utils
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pickle

#https://stats.stackexchange.com/questions/303998/tuning-adaboost
#https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search/notebook
#An Adaboost classifier

def main():

    with open( parent_dir+"/Data/MPII_dataset.p", "rb" ) as f:
        dataset = pickle.load(f)
        num_examples = len(dataset)
        X = np.reshape(dataset, (num_examples, 32))
    with open( parent_dir+"/Data/MPII_dataset_activities.p", "rb" ) as f:
        activities = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_images_names.p", "rb" ) as f:
        images_names = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_label_categories.p", "rb" ) as f:
        categories = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_label.p", "rb" ) as f:
        label = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_euclidean_distance.p", "rb" ) as f:
        distance_euc = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_minkowski_p1.p", "rb" ) as f:
        distance_min_p1 = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_minkowski_p3.p", "rb" ) as f:
        distance_min_p3 = pickle.load(f)
    with open( parent_dir+"/Data/MPII_dataset_cosine.p", "rb" ) as f:
        distance_cos = pickle.load(f)
    
    print("Classyfying dataset = MPIIjoins")

    clf = AdaBoostClassifier(n_estimators=500,learning_rate=0.001)

    title = "Learning Curves (Adaboost)"

    plot_learning_curve(clf, title, X, categories, cv=5)
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
    main()







