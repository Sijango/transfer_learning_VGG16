import os

BASE_PATH = 'images'

TRAIN_PATH = os.path.join(BASE_PATH, 'train')
VALID_PATH = os.path.join(BASE_PATH, 'valid')

BASE_OUTPUT = 'outputs'

MODEL_PATH = os.path.join(BASE_OUTPUT, 'model.h5')
PLOT_PATH = os.path.join(BASE_OUTPUT, 'plot.png')
VALID_FILENAMES = os.path.join(BASE_OUTPUT, 'valid_images.txt')

INIT_LR = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 32
