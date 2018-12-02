
# Load the pickle file.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import utils

def main():
    
    data = utils.load_MPII_data_not_for_cnn(0.7,0,0.3,"cat",True)
    train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y = data

    print("Classyfying...")

    clf = AdaBoostClassifier(n_estimators=400,learning_rate=0.1)
    
    clf.fit(train_set_X,train_set_y)
    print(clf.score(test_set_X, test_set_y))

if __name__ == '__main__':
    main()







