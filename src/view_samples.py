import os
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = os.path.join("dataset", "Training")
CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]

fig, axes = plt.subplots(1, len(CLASSES), figsize=(16, 4))

for ax, class_name in zip(axes, CLASSES):
    class_path = os.path.join(DATASET_DIR, class_name)
    first_image_name = sorted(os.listdir(class_path))[0]
    image_path = os.path.join(class_path, first_image_name)

    img = Image.open(image_path)
    ax.imshow(img, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")

plt.suptitle("One sample MRI per class")
plt.tight_layout()
plt.show()
