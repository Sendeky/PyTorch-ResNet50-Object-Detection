
# import torchvision
import utils.CONFIG as config
# import cv2

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from kitti_dataloader import KittiCustomDataset     # we import our own custom dataloader for the KITTI dataset


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

# def collate_fn(data):
    # img, bbox = data
    # zipped = zip(img, bbox)
    # return list(zipped)

# give image_dir and label_dir to dataloader
trainDS = KittiCustomDataset(image_dir=config.TRAIN_IMAGES_PATH, label_dir=config.TRAIN_LABELS_PATH, transforms=transforms)
# print("trainDS:", trainDS)

# calculate steps per epoch for training
trainSteps = len(trainDS)

# dataloader = DataLoader(trainDS, batch_size=32, shuffle=True)    # dataloder for the training dataset
train_dataloader = DataLoader(trainDS, batch_size=1, shuffle=True)
# Temperorary fix for now is to set batch_size to 1, until collate_fn is ready
# TODO: We need our own collate_fn function to add padding so that we don't get: "RuntimeError: each element in list of batch should be of equal size"

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