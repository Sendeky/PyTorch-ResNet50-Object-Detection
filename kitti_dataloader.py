# import the necessary packages
import numpy
import imageio
import torch
import os
from torch.utils.data import Dataset

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
        
        image = imageio.imread(image_path)
        # label = torch.load(label_path)  # Load your label data
        # label = numpy.loadtxt(label_path)
        with open(label_path, "r") as label_file:
            text = label_file.read()
            print("label file text: ", text)
        # print(f"image{index}", image)
        print(f"label{index}", text)
        # image = self.tensors[0][index]
        # label = self.tensors[1][index]
        # bbox  = self.tensors[2][index]
		# transpose the image such that its channel dimension becomes
		# the leading one
        # image = image.permute(2, 0, 1)
		# check to see if we have any image transformations to apply
		# and if so, apply them
        if self.transforms:
            image = self.transforms(image)
		# return a tuple of the images, labels, and bounding
		# box coordinates
        # return (image, label, bbox)
        return image, text
    
    def __len__(self):
        # return size of dataset
        return len(self.image_files)
    


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