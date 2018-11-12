CURRENT STEP TO COMPLETE:

1) Load the MPII database in MatLab, understand it
2) Transform it to use it with Python

Our python structure:

1) MPII dataset

MPII_dataset = pickle.load( open( "Data/MPII_dataset.p", "rb" ) )

- MPII_dataset is a list that contains the positions of the joints of each example
Each example is a 2x16 matrix where the first row is the x position of the 16 joins and the second row is the y position of the 16 joins. When x and y = 0, it means that the join is not visible

We can match examples with their index in the others pickle files to get the labels for example

Example:
![Alt text](https://github.com/carodak/Human-Movement/blob/master/annotation_example.png "Our structure")
