
# Load the pickle file.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import utils
import os
from time import time
from scipy.stats import randint as sp_randint
from datetime import datetime
from functools import reduce
from sklearn.tree import DecisionTreeClassifier

#https://stats.stackexchange.com/questions/303998/tuning-adaboost
#An Adaboost classifier

def main():
    
    #Dataset would be the coordinates of each join
    data = utils.load_MPII_data_not_for_cnn(0.8,0,0.2,"cat",0)
    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = data
    
    custom_grid_search(train_set_X, train_set_y, test_set_X, test_set_y)
    print("Classyfying...")
    
def custom_grid_search(
                       train_set_x,
                       train_set_y,
                       test_set_x,
                       test_set_y
                       ):
    num_examples = train_set_x.shape[0]

    params = {
        'base_estimator': range(1, 10),
        'n_estimators': range(500, 2000, 100),
        'learning_rate': np.linspace(0.0001,0.01,25)
    }
    
    average_runs_count = 5
    iteration = 1
    total_iterations = reduce(lambda x, y: x*y, [len(lst) for lst in params.values()])
    file_name = "Adaboost_JoinsCoord_Hyperparameters_" + str(int((datetime.utcnow()-datetime(1970,1,1)).total_seconds())) + ".csv"
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", file_name)
    file_results = open(file_path, "w+")
    file_results.write(','.join(["train_accuracy", "test_accuracy", "base_estimator", "n_estimators", "learning_rate", "\n"]))
    for base_estimator_ in params['base_estimator']:
        for n_estimators_ in params['n_estimators']:
            for learning_rate_ in params['learning_rate']:
                    print("Iteration {}/{}".format(iteration, total_iterations))
                    
                    train_accuracy = 0
                    test_accuracy = 0
                    
                    for i in range(average_runs_count):
                        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=base_estimator_),n_estimators=n_estimators_,learning_rate=learning_rate_)
                        model.fit(train_set_x, train_set_y)
                        train_accuracy += model.score(train_set_x, train_set_y)
                        test_accuracy += model.score(test_set_x, test_set_y)
                
                    train_accuracy /= (1.0*average_runs_count)
                    test_accuracy /= (1.0*average_runs_count)
                    
                    file_results.write(','.join(str(x) for x in [train_accuracy, test_accuracy, base_estimator_, n_estimators_, learning_rate_]))
                    file_results.write('\n')
                    iteration += 1

    file_results.close()


if __name__ == '__main__':
    main()







