import os


d_k = 64
d_v = 64
d_ff = 2048
d_model = 160
num_head = 8
num_stacks = 4
num_cameras = 3
num_blocks = 4

LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 1

NUM_FRAMES = 32
RESIZE = 256
CROP = 224
FLIP_FACTOR = 0.5

rate = 0.2    # the proportion of training and validation videos in the raw videos respectively.
remove_mean_image = False
balance = False

path = os.path.dirname(os.getcwd())

DATASET_NAME = 'NTU120'
NUM_CLASSESS = 120
MODEL_NAME = 'model.ckpt'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'Model','checkpoint')
PB_MODEL_SAVE_PATH = os.path.join(path,'Model','pb')
