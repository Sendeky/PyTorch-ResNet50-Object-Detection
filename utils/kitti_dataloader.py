# import the necessary packages
import numpy
import imageio
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from utils.label_encoder import LabelEncoder


class KittiCustomDataset(Dataset):
    # initialize constructor
    def __init__(self, image_dir, annotations, transforms=None):
        self.image_dir  = image_dir
        self.annotations  = annotations
        self.transforms = transforms

        # sorted files
        self.image_files = sorted(os.listdir(image_dir))
        # self.label_files = sorted(os.listdir(label_dir))


    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        image_path = os.path.join(self.image_dir, self.image_files[index])
        text = self.annotations
        # print("len text", len(text))
        # print("index: ", index)
        print("annotations1", text)
        print(f"label index {index}", text[index][0])
        print(f"bbox index {index}", text[index][1])
        if text[index][0]:
            labels = text[index][0]
        if text[index][1]:            
            bboxes = text[index][1]

        print("bboxes: ", bboxes)
        os.wait()
        # label_path = os.path.join(self.label_dir, self.label_files[index])

        # labels = []
        # bboxes = []
        # print("label#: ", index)
        
        # loads the image
        image = imageio.imread(image_path, pilmode="RGB")
        # print("image shape: ", image.shape)

		# check to see if we have any image transformations to apply
		# and if so, apply them
        if self.transforms:
            image = self.transforms(image)

        # we need to one hot encode the labels that the model can perform better
        # first, get unique labels
        unique_labels = set(labels)
        unique_labels = list(unique_labels)
        # print("!!unique_labels ", unique_labels)

        encoded_labels = []
        # iterate through each item in unique_labels and encode them
        for label in unique_labels:
            # print("label: ", label)
            le = LabelEncoder()
            encoded_label = le.KittiLabelEncoder(label)
            encoded_labels.append(encoded_label)

        # print("bboxes: ", bboxes)
        # os.wait()
        labels = torch.from_numpy(np.asarray(encoded_labels))
        ret_bboxes = torch.from_numpy(np.asarray(bboxes))
        # print("encoded_labels:", encoded_labels)

        return image, labels, ret_bboxes
        # return image, encoded_labels, bboxes


    def __len__(self):
        # return size of dataset
        return len(self.annotations)


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