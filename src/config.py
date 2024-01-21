import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 200 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '../Glyphs/train'
# validation images and XML files directory
VALID_DIR = '../Glyphs/test'
CLASSES_DIR = '../Glyphs/classes.txt'

# classes: 0 index is reserved for background
with open(CLASSES_DIR, 'r') as f:
    CLASSES = f.read().splitlines()

# print(CLASSES)
# CLASSES = [
#     'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
# ]
    
NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 50 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 50 # save model after these many epochs