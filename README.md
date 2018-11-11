CURRENT STEP TO COMPLETE:

1) Load the data
2) Understand it

Annotation description

Annotations are stored in a matlab structure RELEASE having following fields
•    .annolist(imgidx) - annotations for image imgidx
    o    .image.name - image filename
    o    .annorect(ridx) - body annotations for a person ridx
        •    .x1, .y1, .x2, .y2 - coordinates of the head rectangle
        •    .scale - person scale w.r.t. 200 px height
        •    .objpos - rough human position in the image
        •    .annopoints.point - person-centric body joint annotations
        •    .x, .y - coordinates of a joint
        •    id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
        •    is_visible - joint visibility
    o    .vidx - video index in video_list
    o    .frame_sec - image position in video, in seconds
•    img_train(imgidx) - training/testing image assignment 
•    single_person(imgidx) - contains rectangle id ridx of sufficiently separated individuals
•    act(imgidx) - activity/category label for image imgidx
    o    act_name - activity name
    o    cat_name - category name
    o    act_id - activity id
•    video_list(videoidx) - specifies video id as is provided by YouTube. To watch video on youtube go to https://www.youtube.com/watch?v=video_list(videoidx)

Example

![Alt text](https://github.com/carodak/Human-Movement/blob/master/annotation_example.png "An example of data loaded")

