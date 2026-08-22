import os

# ----- Paths -----
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# ----- Image settings -----
IMAGE_SIZE = 128        # width and height in pixels (change here if needed)
CHANNELS = 3            # 3 = color (RGB), 1 = grayscale

# ----- Classes -----
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ----- Training settings (used later in Stage 7) -----
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ----- Validation split -----
VALIDATION_SPLIT = 0.2   # 20% of Training data held out for validation
