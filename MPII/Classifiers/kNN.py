from mat4py import loadmat
import numpy as np
import json
import pickle




#Import the data
json1_file = open('trainval.json')
json1_str = json1_file.read()
json1_data = json.loads(json1_str)


# Initialize list to handle dataset
joints_dataset = [None]*len(json1_data)
data_image_name = [None]*len(json1_data)



for i in range(len(json1_data)):

    joints = json1_data[i]['joints']
    centers = json1_data[i]['center']
    scale_factor = json1_data[i]['scale']
    joints_vis = json1_data[i]['joints_vis']
    data_image_name[i] = json1_data[i]['image']

    #Convert data to numpy arrays before putting them in the list
    joints = np.array(joints)
    centers = np.array(centers)

    #Center and scale the images
    joints = joints - centers
    joints = joints * scale_factor

    # Convert data to int type
    joints = joints.astype(int)

    # Putting to 0 the non-visible joints coordinates
    joints_dataset[i] = joints.T * joints_vis

# Loading the image name vector to match with labels
label_image_names = loadmat('image_file.mat')

# Simplify the access to labels names
label_image_names = label_image_names['annolist']
label_image_names = label_image_names['image']

for i in range(len(label_image_names)):
    label_image_names[i] = label_image_names[i]['name']


# Initialize a list that will match the indices of the labels' vector for every example
data_label_match = [None]*len(json1_data)



for i in range(len(json1_data)):

    try:
        data_label_match[i] = label_image_names.index(data_image_name[i])
    except IndexError:
        data_label_match[i] = 'sorry, no label'






labels_list = loadmat('act.mat')


labels = labels_list['act']


labels_id = labels['act_id']
labels_act = labels['act_name']
labels_cat = labels['cat_name']


data_label_id = [None]*len(json1_data)
data_label_act = [None]*len(json1_data)
data_label_cat = [None]*len(json1_data)

for i in range(len(json1_data)):

    try:
        data_label_id[i] = labels_id[data_label_match[i]]
        data_label_act[i] = labels_act[data_label_match[i]]
        data_label_cat[i] = labels_cat[data_label_match[i]]

    except IndexError:
        print ('FUCK')


data_matrix = np.zeros((32,len(joints_dataset)))

for i in range(len(joints_dataset)):

    test = np.hstack((joints_dataset[i][0,:], joints_dataset[i][1,:]))
    data_matrix[:,i] = test



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data_label_id = np.array(data_label_id)


X_train, X_test, y_train, y_test = train_test_split(data_matrix.T, data_label_id, test_size=0.01)

knn = KNeighborsClassifier(n_neighbors=80)
knn.fit(X_train, y_train)
val = knn.predict(X_test)

h = 0
same =[0]
for i in range(len(val)):
    if(val[i] == y_test[i]):
        h +=1

k = 0
same = np.zeros(h)
for i in range(len(val)):
    if(val[i] == y_test[i]):
        same[k] = val[i]
        k += 1


for i in range(len(same)):
    t = np.where(data_label_id == same[i])


    print (data_label_act[t[0][0]])
    print(data_label_cat[t[0][0]])






# conf_matrix = functions.conf_matrix(y_test, val)


# Save into a pickle file.
pickle.dump( joints_dataset, open( "MPII_dataset.p", "wb" ) )

pickle.dump(data_label_act, open( "MPII_dataset_activities.p", "wb" ) )

pickle.dump(data_label_id, open( "MPII_dataset_label.p", "wb" ) )

pickle.dump( data_label_cat, open( "MPII_dataset_label_categories.p", "wb" ) )

pickle.dump( data_image_name, open( "MPII_dataset_images_names.p", "wb" ) )



tt = np.zeros((len(data_label_id), data_label_id.max()+1))

for i in range(len(data_label_id)):
    if (data_label_id[i] == -1):
        tt[i, 0] = 1

    else:
        tt[i, data_label_id[i]] = 1

    if (data_label_id[i] == 5):
        p_n = (data_image_name[i])
        print(data_label_cat[i])
        print(data_label_act[i])



mask = np.ones(data_label_id.max()+1, dtype=bool)
mask = np.sum(tt, 0)>0

result = tt[:, mask]



bob = 0

