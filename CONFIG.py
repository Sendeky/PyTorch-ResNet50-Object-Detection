# import the necessary packages
import torch
import os


# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "./dataset"
TRAIN_IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
TRAIN_LABELS_PATH = os.path.sep.join([BASE_PATH, "label_2"])
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])