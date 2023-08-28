import torchvision
import CONFIG as config
# import cv2



from torchvision import transforms
from torchvision.datasets import ImageFolder
from kitti_dataloader import KittiCustomDataset     # we import our own custom dataloader for the KITTI dataset


# variables for the images, labels, and bboxes
data = []
labels = []
bboxes = []
imagePaths = []

# get tuple of images, labels, bbxoxes into custom dataset

# 
transforms = transforms.Compose([
    transforms.ToPILImage(),
	transforms.ToTensor(),
	# transforms.Normalize(mean=config.MEAN, std=config.STD)    # get mean and standard deviation of image pixel values
])
# give image_dir and label_dir to dataloader
trainDS = KittiCustomDataset(image_dir="./dataset/train/image_2/", label_dir="./dataset/train/label_2/", transforms=transforms)

# image = ImageFolder(root=config.TRAIN_IMAGES_PATH)

