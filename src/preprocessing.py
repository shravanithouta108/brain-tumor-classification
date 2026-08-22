import os
import numpy as np
import cv2
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, CLASS_NAMES


def load_dataset(folder_path):
    """
    Loads all images from a dataset folder (Training or Testing),
    resizes them, normalizes pixel values, and returns:
    - images as a NumPy array
    - labels as a NumPy array (numbers, matching CLASS_NAMES order)
    """
    images = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(folder_path, class_name)
        filenames = os.listdir(class_folder)

        for filename in filenames:
            filepath = os.path.join(class_folder, filename)

            # Read image (OpenCV loads in BGR color order by default)
            img = cv2.imread(filepath)
            if img is None:
                continue  # skip unreadable files, just in case

            # Resize to our fixed size
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Convert BGR (OpenCV default) to RGB (standard order)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(class_index)

    # Convert lists to NumPy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")

    # Normalize pixel values from 0-255 range to 0-1 range
    images = images / 255.0

    return images, labels


if __name__ == "__main__":
    print("Loading training data...")
    X_train, y_train = load_dataset(TRAIN_DIR)
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    print("\nLoading testing data...")
    X_test, y_test = load_dataset(TEST_DIR)
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")
