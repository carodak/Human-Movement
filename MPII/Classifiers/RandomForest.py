import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("target")
arg_parser.add_argument('-d',
        dest='use_dist', type=bool)

def run(args):
    path = os.path.join('..','Data')
    with open(os.path.join(path, 'MPII_dataset.p'),'rb') as f:
        dataset = pickle.load(f)
    with open(os.path.join(path, 'MPII_dataset_activities.p'),'rb') as f:
        activities = pickle.load(f)
    with open(os.path.join(path, 'MPII_dataset_images_names.p'),'rb') as f:
        images_names = pickle.load(f)
    with open(os.path.join(path, 'MPII_dataset_label_categories.p'),'rb') as f:
        categories = pickle.load(f)
    with open(os.path.join(path, 'MPII_dataset_label.p'),'rb') as f:
        label = pickle.load(f)
    with open(os.path.join(path, 'MPII_dataset_euclidean_distance.p'),'rb') as f:
        distance = pickle.load(f)

    if args.target.startswith('a'):
        Y = categories
    else:
        Y = activities

    if args.use_dist:
        n = distance.shape[0]
        X = np.reshape(distance, (n, 16*16))
    else: 
        n = len(dataset)
        X = np.reshape(dataset, (n, 32))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    model = RandomForestClassifier(max_depth=20)
    model.fit(x_train, y_train)

    print(model.score(x_test, y_test))

if __name__=='__main__':
    args = arg_parser.parse_args()
    run(args)
