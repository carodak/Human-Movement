import numpy as np
import json
import pickle

json_filelist = pickle.load( open( "json_filelist.p", "rb" ) )
action_label = pickle.load( open( "action_label.p", "rb" ) )
file_names = pickle.load( open( "file_names.p", "rb" ) )


pose_keypoints = [None] * len(json_filelist)
joints_pos = [None] * len(json_filelist)
no_detect = []

for i in range(len(json_filelist)):

    json1_file = open(json_filelist[i])
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)
    people = json1_data['people']
    try :
        pose_keypoints[i] = people[0]['pose_keypoints']

        pose_keypoints[i] = np.array(pose_keypoints[i])

        joints = [None]*18
        joint_ind = 0

        for k in range(int(len(pose_keypoints[i])/3)):
                joints[joint_ind] = pose_keypoints[i][3 * k: 3 * k+3]
                joint_ind += 1
        joints = np.array(joints)

        joints_pos[i] = np.array(joints)

    except:
        joints_pos[i] = np.zeros((18,3))
        no_detect.append(i)

no_detect = np.array(no_detect)

idx = 0

joints_pose_keypoints = [None] * (len(joints_pos)-len(no_detect))

labels = [None] * (len(joints_pos)-len(no_detect))

picture_name = [None] * (len(joints_pos)-len(no_detect))



pickle.dump(no_detect, open( "no_detect_idx.p", "wb" ) )






for i in range(len(joints_pos)):

    joints_pose_keypoints[i- idx] = joints_pos[i]

    labels[i-idx] = action_label[i]

    picture_name[i-idx] = file_names[i]

    if idx < len(no_detect):
        if i == no_detect[idx]:
            idx +=1

    if idx == 133:
        call_bob = 0





pickle.dump(joints_pose_keypoints, open( "stanford_joints_pose.p", "wb" ) )
pickle.dump(labels, open( "stanford_labels.p", "wb" ) )
pickle.dump(picture_name, open('stanford_picture_name.p','wb'))








bob = 0