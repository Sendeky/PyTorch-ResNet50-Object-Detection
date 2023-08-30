# import the necessary packages
import numpy
import imageio
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class KittiCustomDataset(Dataset):
    # initialize constructor
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        self.transforms = transforms

        # sorted files
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))


    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        image_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        labels = []
        bboxes = []
        print("label#: ", index)
        
        # loads the image
        image = imageio.imread(image_path)
        print("image shape: ", image.shape)

        # loads the text file with the labels
        with open(label_path, "r") as label_file:
            text = label_file.read()
            text_arr = text.splitlines() # splits text into array by newlines (some labels have more than one object)

            for line in text_arr:
                split_line = line.split()         # splits line by whitespace so we can extract words/numbers

                label = split_line[0]      # label is 0th item in text
                # print("labels22", nparr)
                # perform label encoding on the labels
                # le = LabelEncoder()
                # label = le.fit_transform(nparr)
                # print("labels", label)
                labels.append(label)
                # print("labels", labels)
                bboxes.append(split_line[4:8])    # bounding boxes are 4th - 8th items in text  

            print("label file text: ", text)
            print("labels: ", labels)
            print("bboxes: ", bboxes)
        # print(f"image{index}", image)
        # print(f"label arr {index}", labels)

		# check to see if we have any image transformations to apply
		# and if so, apply them
        if self.transforms:
            image = self.transforms(image)

        # we need to one hot encode the labels that the model can perform better
        # first, get unique labels
        unique_labels = set(labels)
        unique_labels = list(unique_labels)
        # then one hot encode them with sklearn LabelEncoder
        print("unique labels: ", unique_labels)
        le = LabelEncoder()
        labels = le.fit_transform(unique_labels)
        print("encoded labels:", labels)

        # returns image, labels, and boundingboxes
        return image, labels, bboxes
    
    def __len__(self):
        # return size of dataset
        return len(self.image_files)
    
## DATALOADER Needs to be rewritten to accept a tensor of read labels
## this is so we can do the one hot encoding for the labels which
## improves accuracy

# label.txt format for KITTI dataset
"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""