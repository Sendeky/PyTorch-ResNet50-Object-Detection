# import the necessary packages
# import torch
import os
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "./dataset"
TRAIN_IMAGES_PATH = os.path.sep.join([BASE_PATH, "/train/image_2/"])
TRAIN_LABELS_PATH = os.path.sep.join([BASE_PATH, "/train/label_2/"])
TRAIN_LENGTH = 7480
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# these are standard deviation and Mean from ImageNet dataset
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]

# learning rate
INIT_LR = 0.002

# epochs
NUM_EPOCHS = 20

# bbox and label class loss
BBOX = 1.0
LABELS = 1.0