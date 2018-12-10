import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import json
import pickle



def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()






xml_filelist = glob.glob('/home/ncls/Documents/IFT6390/Projet/database/Stanford/XMLAnnotations/*.xml')

file_names = [None]*len(xml_filelist)
action_label = [None]*len(xml_filelist)
bndbox = np.zeros((len(xml_filelist),4)).astype(int)
image_filelist = [None]*len(xml_filelist)
json_filelist = [None]*len(xml_filelist)

for i in range(len(xml_filelist)):
    tree = ET.parse(xml_filelist[i])
    root = tree.getroot()

    file_names[i] = root[0].text

    image_filelist[i] = '/home/ncls/Documents/IFT6390/Projet/database/Stanford/JPEGImages/' + file_names[i]

    action_label[i] = root[2][1].text


    bndbox[i, 0] = int(root[2][2][0].text)       #X_max
    bndbox[i, 1] = int(root[2][2][1].text)       #X_min
    bndbox[i, 2] = int(root[2][2][2].text)       #Y_max
    bndbox[i, 3] = int(root[2][2][3].text)          #Y_min


    # crop_name = '/home/ncls/Documents/IFT6390/Projet/database/Stanford/cropped_images/1/'+file_names[i]
    # crop(image_filelist[i], (bndbox[i, 1], bndbox[i, 3], bndbox[i, 0], bndbox[i, 2]), crop_name)

    json_filelist[i] = '/home/ncls/Documents/IFT6390/Projet/database/Stanford/OpenPose_annotation/1_no_hands/' + file_names[i][:-4] + '_keypoints.json'

pickle.dump(bndbox, open( "bndbox.p", "wb" ) )

pickle.dump(json_filelist, open( "json_filelist.p", "wb" ) )
pickle.dump(action_label, open( "action_label.p", "wb" ) )
pickle.dump(file_names, open( "file_names.p", "wb" ) )




bob = 0