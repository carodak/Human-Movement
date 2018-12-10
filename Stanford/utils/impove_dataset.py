import numpy as np
import pickle
import glob


xml_filelist = glob.glob('/home/ncls/Documents/IFT6390/Projet/database/Stanford/XMLAnnotations/*.xml')


# bndbox = pickle.load( open( "../Data/bndbox.p", "rb"  ))

bnd_box = pickle.load( open( "../Data/bndbox.p", "rb" ) )

no_detect = pickle.load( open( "../Data/no_detect_idx.p", "rb" ) )

joint_pose = pickle.load( open( "../Data/stanford_joints_pose.p", "rb" ) )

label_picture_names = pickle.load( open( "../Data/stanford_picture_name.p", "rb" ) )

labels = pickle.load( open( "../Data/stanford_labels.p", "rb" ) )



bnd_box = np.delete(bnd_box, no_detect, axis=0)

x_length = [None] * len(bnd_box)
y_length = [None] * len(bnd_box)


j=0

for i in range(len(bnd_box)):

    x_length[i] = bnd_box[i,0] - bnd_box[i,1]

    y_length[i] = bnd_box[i,2] - bnd_box[i,3]






bob = 0