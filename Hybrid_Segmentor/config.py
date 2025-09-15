import torch
import os 

### train on total dataset
# NUM_EPOCHS = 1000
DATASET_SIZE = {'train' : 9600, 'val' : 1200, 'test' : 1200}
# dataset = os.path.join('../', 'split_dataset_final/')

### train on total dataset
NUM_EPOCHS = 100
# DATASET_SIZE = {'train' : 192, 'val' : 24, 'test' : 24} # masonry
# DATASET_SIZE = {'train' : 3484, 'val' : 435, 'test' : 436} # steelcrack
# DATASET_SIZE = {'train' : 3782, 'val' : 472, 'test' : 474} # pavement

# dataset = os.path.join('../', 'sample_dataset/') # or split_dataset_final
dataset = "/kaggle/input/crackvision12k/split_dataset_final/"
# dataset = "/kaggle/input/masonry/MASONRY/"
# dataset = "/kaggle/input/steels/STEELS/"
# dataset = "/kaggle/input/pavement/PAVEMENT/"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

# Dataset dir
TRAIN_IMG_DIR = dataset+"train/IMG"
TRAIN_MASK_DIR = dataset+"train/GT"
VAL_IMG_DIR = dataset+"val/IMG"
VAL_MASK_DIR = dataset+"val/GT"
TEST_IMG_DIR = dataset+"test/IMG"
TEST_MASK_DIR = dataset+"test/GT"

# checkpoint
CHECKPOINTS_PATH = "/kaggle/input/hybrid-checkpoints/checkpoints/hybrid_segmentor_BCE_2.ckpt"
