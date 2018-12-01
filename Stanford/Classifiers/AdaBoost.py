# Load the pickle file.
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def main():

    parent_dir = dirname(dirname(abspath(__file__)))
    parent_parent_dir = dirname(parent_dir)

    #Load the data
    dataset = pickle.load( open( parent_parent_dir+"/Stanford/Data/stanford_joints_pose.p", "rb" ) )
    labels = pickle.load( open( parent_parent_dir+"/Stanford/Data/stanford_labels.p", "rb" ) )

    #print(dataset)
    #print(dataset.shape)

    #Change the list to an array
    X = np.array([x.tolist() for x in dataset])
    X = X[:,:,:-1]
    print(X.shape)

    n = X.shape[0]

    X = np.reshape(X, (n, 36))

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


    clf = AdaBoostClassifier(n_estimators=500,learning_rate=0.1)
    clf.fit(X_train,y_train)

    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()






