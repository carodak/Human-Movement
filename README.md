CURRENT STEP:
Start classyfying

PAST STEP:

1) Load the MPII database in MatLab, understand it (done)
2) Transform it to use it with Python (done)

Our python structure:

1) MPII dataset

MPII_dataset = pickle.load( open( "Data/MPII_dataset.p", "rb" ) )

- MPII_dataset is a list that contains the positions of the joints of each example
Each example is a 2x16 matrix where the first row is the x position of the 16 joins and the second row is the y position of the 16 joins. When x and y = 0, it means that the join is not visible

![Alt text](https://github.com/carodak/Human-Movement/blob/master/annotation_example.png "Our structure")

How to use it: Match examples with their index in the others pickle files to get the labels...

