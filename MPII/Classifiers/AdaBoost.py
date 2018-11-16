# Load the pickle file.
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

#Load the data
MPII_dataset = pickle.load( open( "Data/MPII/MPII_dataset.p", "rb" ) )
MPII_dataset_activities = pickle.load( open( "Data/MPII/MPII_dataset_activities.p", "rb" ) )
MPII_dataset_images_names = pickle.load( open( "Data/MPII/MPII_dataset_images_names.p", "rb" ) )
MPII_dataset_label_categories = pickle.load( open( "Data/MPII/MPII_dataset_label_categories.p", "rb" ) )
MPII_dataset_label = pickle.load( open( "Data/MPII/MPII_dataset_label.p", "rb" ) )

#Our features would be the x,y coordinates of the 16 skeleton joins on each image: MPII_dataset
#meaning that x = (join1.x,join2.x,...,join16.x,join1.y,join2.y...,join32.y)
#Our target would be the label of activities: MPII_dataset_label

#Get the number of images
n = len(MPII_dataset)

#We need the x_data under the form [[32joins of image1], [32joins of image2], []...]
data = np.reshape(MPII_dataset, (n, 32))

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








