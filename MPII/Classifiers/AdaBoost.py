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
    data = utils.load_MPII_data_not_for_cnn(0.8,0,0.2,"cat",1)
    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = data
    
    print("Classyfying...")
    clf = AdaBoostClassifier(n_estimators=500,learning_rate=0.001)

    clf.fit(train_set_X,train_set_y)

    print("Score on training set: ",clf.score(train_set_X,train_set_y),"Score on testing set: ", clf.score(test_set_X,test_set_y))


if __name__ == '__main__':
    main()





