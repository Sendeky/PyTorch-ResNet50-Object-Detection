
# import torchvision
import utils.CONFIG as config
from torchvision.datasets import ImageFolder

import os
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.kitti_dataloader import KittiCustomDataset     # we import our own custom dataloader for the KITTI dataset


# variables for the images, labels, and bboxes
data = []
labels = []
bboxes = []
imagePaths = []

# get tuple of images, labels, bbxoxes into custom dataset

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
            temp_annot = []
            temp_labels = []
            temp_bboxes = []

            for line in text_arr:
                split_line = line.split()         # splits line by whitespace so we can extract words/numbers

                label = split_line[0]      # label is 0th item in text
                temp_labels.append(label)
                # print("labels", labels)
                temp_bboxes.append(split_line[4:8])    # bounding boxes are 4th - 8th items in text  

            # combine array of labels and array of boxes into temp_annot array (will be used as tensor for porcessing later)
            temp_annot = [temp_labels, temp_bboxes]
            labels.append(temp_annot)

print("len labels", len(labels))
print("labels", labels[145])
# os.wait()
# print("shape labels", labels.shape)

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

for images, labels, bboxes in train_dataloader:
	print("")
	# print("labels", labels)

# load the ResNet50 network
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)     # we use default ResNet50 weights (pretrained=True param will be deprecated)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
	param.requires_grad = False
	
# create our custom object detector model and flash it to the current
# device
# objectDetector = ObjectDetector(resnet, len(le.classes_))
# objectDetector = objectDetector.to(config.DEVICE)