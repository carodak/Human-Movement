# Load the pickle file.
import pickle
MPII_dataset = pickle.load( open( "Data/MPII_dataset.p", "rb" ) )
#joint information stack in a vector form.  the first 16 values are the x value of each joints and the next 16 is the y value.  When both x and y are 0 that means that the information wasn't available

print(MPII_dataset)

label_categories = pickle.load( open( "Data/MPII_dataset_label_categories.p", "rb" ) )

#print(label_categories)
