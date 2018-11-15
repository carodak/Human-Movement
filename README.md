## CURRENT STEP: Start classyfying MPII dataset

I) Our python structure:

The features have been separated into several pickles. For each example, we have to use the different pickles files to get its features.

#### **1) MPII_dataset.p**

*MPII_dataset = pickle.load( open( "Data/MPII_dataset.p", "rb" ) )*

MPII_dataset is a list that contains the positions of the joins of each image.
Each example is a 2x16 matrix where the first row is the x position of the 16 joins and the second row is the y position of the 16 joins. When x and y = 0, it means that the join is not visible

***Example:*** 
MPII_dataset[0] = x,y coordinates of 16 joins of the first image  

=  [ [ 115  107  -85  -60  120  130  -72  -89  -84  -36   64  -50 -101  -79 -25   56] [ 280  105   97   72   97  278   85 -107 -118 -235   74   56 -101 -115 31   58] ]

#### **2) MPII_dataset_activities.p**

*MPII_dataset_activities = pickle.load(open("Data/MPII/MPII_dataset_activities.p", "rb" ) )*

MPII_dataset_activities is a vector that gives the name of activity(ies) of each image

***Example:*** 
MPII_dataset_activities[0] = 'curling'

MPII_dataset_activities[17] = 'sitting, talking in person, on the phone, computer, or text messaging, light effort'

#### **3) MPII_dataset_images_names.p**

*MPII_dataset_images_names = pickle.load( open( "Data/MPII/MPII_dataset_images_names.p", "rb" ) )*

MPII_dataset_images_names is a vector that gives the name of each image

***Example:***
MPII_dataset_images_names[0]='015601864.jpg'

#### **4) MPII_dataset_label_categories.p**

*MPII_dataset_label_categories = pickle.load( open( "Data/MPII/MPII_dataset_label_categories.p", "rb" ) )*

MPII_dataset_label_categories is a vector that gives the category of each image

***Example:***
MPII_dataset_label_categories[0]='sports'

#### **5)  MPII_dataset_label.p**

*MPII_dataset_label = pickle.load( open( "Data/MPII/MPII_dataset_label.p", "rb" ) )*

MPII_dataset_label is a vector that gives the label of the activity/activities

***Example:***
MPII_dataset_label[17]=622

#### **6)  Example with one image**

Let's take picture number 17.

**The name of the picture 17 is:** MPII_dataset_images_names[17]
**Coordinates of its joins :** MPII_dataset[17]
**Activity/Activities:** MPII_dataset_activities[17]
**Category:** MPII_dataset_label_categories[17]
**Label of the activity/activities:** MPII_dataset_label[17]

it gives:

The name of the picture 17 is 018340451.jpg

Coordinates of its joins  :  [[ 115  107  -85  -60  120  130  -72  -89  -84  -36   64  -50 -101  -79
-25   56][ 280  105   97   72   97  278   85 -107 -118 -235   74   56 -101 -115
31   58]]

Activity/Activities:  sitting, talking in person, on the phone, computer, or text messaging, light effort

Category:  miscellaneous

Label of the activity/activities:  622

**Show the picture:**

Notice that you got the name of the picture so you can see it via mpi website: 
http://human-pose.mpi-inf.mpg.de/js_viewer/thumbs/018340451_thumb.jpg?fbclid=IwAR3n_eimmS4rKQVmhubcjivP9O3j1OfW9FwWq9T1fvGwR6dwGftaKK1zuKk
