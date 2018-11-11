import scipy.io
import numpy as np

data = scipy.io.loadmat("Data/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")

for i in data:
    print("i: %s" % i)
    if '__' not in i and 'readme' not in i:
        np.savetxt(("Data/CSV/"+i+".csv"),data[i],delimiter=',')
