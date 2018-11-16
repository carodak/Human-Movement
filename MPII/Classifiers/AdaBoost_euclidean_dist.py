# Load the pickle file.
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from os.path import dirname, abspath

current_dir = dirname(dirname(abspath(__file__)))
parent_dir = dirname(current_dir)

#Load the data
MPII_dataset_euclidean_distance = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_euclidean_distance.p", "rb" ) )
MPII_dataset_activities = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_activities.p", "rb" ) )
MPII_dataset_images_names = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_images_names.p", "rb" ) )
MPII_dataset_label_categories = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_label_categories.p", "rb" ) )
MPII_dataset_label = pickle.load( open( parent_dir+"/MPII/Data/MPII_dataset_label.p", "rb" ) )

#Our features would be the euclidean distance
#Our target would be the label of activities: MPII_dataset_label

#Get the number of images
n = len(MPII_dataset_euclidean_distance)

#We need the x_data under the form [[32joins of image1], [32joins of image2], []...]
data = np.reshape(MPII_dataset_euclidean_distance, (n, 256))
print(data)

#We just take a subset of the data to train -> 100 first images
x_train = np.delete(data, slice(1001, n), 0)
y_train = np.delete(MPII_dataset_label, slice(1001,n),0)

#Let's take 10 weak learners
clf = AdaBoostClassifier(n_estimators=10)

#cv : Determines the cross-validation splitting strategy
scores = cross_val_score(clf, x_train, y_train, cv=5)

mean = scores.mean()
print("Scores: ",scores)
print("Mean: ",mean)








