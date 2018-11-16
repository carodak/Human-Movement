# To load pickle files
import pickle


def main():
    MPII_dataset = pickle.load(open("Data/MPII_dataset.p", "rb"))

    # MPII_dataset: list that contains all examples (=annotation of all pictures)

    # Each example is a 2x16 matrix where the first row is the x position of the 16 joints
    # and the second row is the y position of the 16 joints.
    # When x and y = 0, it means that the joint is not visible

    # Then we can match examples with their index in the others pickle files to get the labels for example

    MPII_dataset_activities = pickle.load(open("Data/MPII_dataset_activities.p", "rb"))
    MPII_dataset_images_names = pickle.load(open("Data/MPII_dataset_images_names.p", "rb"))
    MPII_dataset_label_categories = pickle.load(open("Data/MPII_dataset_label_categories.p", "rb"))
    MPII_dataset_label = pickle.load(open("Data/MPII_dataset_label.p", "rb"))

    example = 17
    print("The name of the picture %s is %s " % (example,MPII_dataset_images_names[example]))
    print("Coordinates of its joins  : ",MPII_dataset[example])
    print("Activity/Activities: ",MPII_dataset_activities[example])
    print("Category: ",MPII_dataset_label_categories[example])
    print("Labels of activity/activities: ",MPII_dataset_label[example])


if __name__ == '__main__':
    main()
