# Load the pickle file.
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    main()

    current_dir = dirname(dirname(abspath(__file__)))
    parent_dir = dirname(current_dir)

    #Load the data
    MPII_dataset = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset.p", "rb" ) )
    MPII_dataset_images_names = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_images_names.p", "rb" ) )
    MPII_dataset_label_categories = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_label_categories.p", "rb" ) )
    MPII_dataset_label = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_label.p", "rb" ) )

    #Our features would be the x,y coordinates of the 16 skeleton joins on each image: MPII_dataset
    #meaning that x = (join1.x,join2.x,...,join16.x,join1.y,join2.y...,join32.y)
    #Our target would be the label of activities: MPII_dataset_label

    #Get the number of images
    n = len(MPII_dataset)

    #We need the x_data under the form [[32joins of image1], [32joins of image2], []...]
    X = np.reshape(MPII_dataset, (n, 32))

    X_train, X_test, y_train, y_test = train_test_split(X, MPII_dataset_label_categories, test_size=0.33, random_state=42)

    clf = AdaBoostClassifier(n_estimators=300,learning_rate=0.0001)

    clf.fit(X_train,y_train)
    print(clf.score(X_test, y_test))

if __name__ == '__main__':
    main()







