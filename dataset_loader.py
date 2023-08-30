
# import torchvision
import CONFIG as config
# import cv2

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
print("trainDS:", trainDS)
dataloader = DataLoader(trainDS, batch_size=32, shuffle=True)    # dataloder for the training dataset
# Temperorary fix for now is to set batch_size to 1, until collate_fn is ready
# TODO: We need our own collate_fn function to add padding so that we don't get: "RuntimeError: each element in list of batch should be of equal size"

# iterates through the images, labels, and  bboxes in dataloader
for images, labels, bboxes in dataloader:
    print("image:", images)
    print("labels:", labels)

# image = ImageFolder(root=config.TRAIN_IMAGES_PATH)

