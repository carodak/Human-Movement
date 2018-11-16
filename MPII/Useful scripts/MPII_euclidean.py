# Save a dictionary into a pickle file.
import pickle
import numpy as np
from scipy.spatial import distance

#Load the data
MPII_dataset = pickle.load( open( "../Data/MPII/MPII_dataset.p", "rb" ) )

#Get the number of images
n = len(MPII_dataset)

#Number of joins
n1 = len(MPII_dataset[0][0])

#Distance between two joins
distP1P2 = np.array([])

#dist_i: would be a 16x16 matrix: distance from a join to the others ones
#[ [dist(join1,join1),dist(join1,join2),...,dist(join1,join16)] , ...,[dist(join16,join1),dist(join16,join2),...,dist(join16,join16)]
# not optimized
dist_i = np.array([]).reshape(0,16)

#dist: is the final matrix (dist_i for all pictures)
dist = np.zeros([n,16,16])

#Compute the euclidean distances
for i in range(n):
    
    for j in range(n1):
        
        x_coordP1 = MPII_dataset[i][0][j]
        y_coordP1 = MPII_dataset[i][1][j]
        
        distP1P2 = np.array([])
        
        for h in range(n1):
            x_coordP2 = MPII_dataset[i][0][h]
            y_coordP2 = MPII_dataset[i][1][h]
            distP1_P2 = distance.euclidean([x_coordP1,y_coordP1], [x_coordP2,y_coordP2])
            distP1P2 =  np.append(distP1P2,distP1_P2)
        
        dist_i = np.vstack((dist_i, distP1P2))
#print(dist_i)
    dist[i] = dist_i
    dist_i = np.array([]).reshape(0,16)

print(dist)
pickle.dump(dist, open( "Data/MPII/MPII_dataset_euclidean_distance.p", "wb" ) )












