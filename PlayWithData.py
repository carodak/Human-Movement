# Load the pickle file.
import pickle
MPII_dataset = pickle.load( open( "Data/MPII_dataset.p", "rb" ) )

#MPII_dataset: list that contains all examples (=annotation of all pictures)

#Each example is a 2x16 matrix where the first row is the x position of the 16 joins and the second row is the y position of the 16 joins
#When x and y = 0, it means that the join is not visible

#Then we can match examples with their index in the others pickle files to get the labels for example


print(MPII_dataset)

