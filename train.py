import os
import numpy as np
import time
import torch

from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# our utils
import utils.CONFIG as config
from utils.bbxo_regressor import ObjectDetector
from utils.kitti_dataloader import KittiCustomDataset     # we import our own custom dataloader for the KITTI dataset
from utils.label_encoder import LabelEncoder


# variables for the images, labels, and bboxes
data = []
labels = []
bboxes = []
imagePaths = []


# transforms for normalizing images (model needs tensor + normalization + same dimensions)
transforms = transforms.Compose([
    transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD),    # get mean and standard deviation of image pixel values
    transforms.Resize((640, 480))
])


## cycle through labels, put them in a tensor,
## and give tensor to dataloader
for index in range(config.TRAIN_LENGTH):
    label_files = sorted(os.listdir(config.TRAIN_LABELS_PATH))              # dir with labels
    label_path = os.path.join(config.TRAIN_LABELS_PATH, label_files[index]) # path of label with index
	
    # loads text file and gets the labels
    with open(label_path, "r") as label_file:
            text = label_file.read()
            text_arr = text.splitlines() # splits text into array by newlines (some labels have more than one object)

            # variable that stores the labels from the txt (there can be several labels per txt)
            # temp_annot = []
            annot_labels = []
            annot_bboxes = []

            for line in text_arr:
                split_line = line.split()         # splits line by whitespace so we can extract words/numbers

                label = split_line[0]      # label is 0th item in text
                annot_labels.append(label)
                # print("labels", labels)

                set_bboxes = []     # set of 4 bbox coordinates (needed to correctly parse bboxes)
                # annot_box_float = float(split_line[4:8])    # bounding boxes are 4th - 8th items in text  
                for bbox_coord in split_line[4:8]:
                      annot_box_float = float(bbox_coord)   # we need float for bboxes instead of string
                      set_bboxes.append(annot_box_float)    # append the float bbox coordinates to our set
                      
                annot_bboxes.append(set_bboxes)
                # annot_bboxes.append(split_line[4:8])    

            # combine array of labels and array of boxes into temp_annot array (will be used as tensor for porcessing later)
            # os.wait()
            temp_annot = [annot_labels, annot_bboxes]
            labels.append(temp_annot)

print("len labels", len(labels))
# print("labels", labels[145]

# def collate_fn(data):
    # img, bbox = data
    # zipped = zip(img, bbox)
    # return list(zipped)

# give image_dir and label_dir to dataloader
trainDS = KittiCustomDataset(image_dir=config.TRAIN_IMAGES_PATH, annotations=labels, transforms=transforms)
# print("trainDS:", trainDS)

# calculate steps per epoch for training
trainSteps = len(trainDS)

# dataloader = DataLoader(trainDS, batch_size=32, shuffle=True)    # dataloder for the training dataset
train_dataloader = DataLoader(trainDS, batch_size=1, shuffle=True)
# Temperorary fix for now is to set batch_size to 1, until collate_fn is ready
# TODO: We need our own collate_fn function to add padding so that we don't get: "RuntimeError: each element in list of batch should be of equal size"

# for images, labels, bboxes in train_dataloader:
	# print("")


# load the ResNet50 network
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)     # we use default ResNet50 weights (pretrained=True param will be deprecated)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
	param.requires_grad = False

le = LabelEncoder()
# create our custom object detector model and move it to the current device
print("len classes: ", le.len_classes())
objectDetector = ObjectDetector(resnet, le.len_classes())
objectDetector = objectDetector.to(config.DEVICE)
	
# loss functions for classification and bbox detection
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

# initializer optimizer
opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
print(objectDetector)

# dictionary for training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
	 "val_class_acc": []}


# loop over epochs for training
print("[INFO] training the network...")
startTime = time.time()

# tqdm is simply a progress bar 
for epoch in tqdm(range(config.NUM_EPOCHS)):
      # set nueral net to training mode
      objectDetector.train()
      
      # initialize the total training and validation loss
      totalTrainLoss = 0
      totalValLoss = 0
      
      # initialize the number of correct predictions in the training
      # and validation step
      trainCorrect = 0
      valCorrect = 0
      
      # loop over images in training set
      for (images, labels, bboxes) in train_dataloader:
            # send input to device
            (images, labels, bboxes) = (images.to(config.DEVICE),
			labels.to(config.DEVICE), bboxes.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            predictions = objectDetector(images)

            print("bboxLoss shape: ", bboxes.shape)
            print("classLoss label shape: ", labels.shape)
            # get loss for bboxes and labels
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            print("labels shape: ", labels.shape)
            print("labels: ", labels)
            print("labels0: ", labels[0])
            classLoss = classLossFunc(predictions[1], labels[0])
            totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)

            # Zero gradients, perform backprogation 
            # and update weights
            opt.zero_grad()
            totalLoss.backward()
            opt.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += totalLoss
            trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()