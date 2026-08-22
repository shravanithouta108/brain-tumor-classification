import os
from PIL import Image

DATASET_DIR = "dataset"

def inspect_split(split_name):
    split_path = os.path.join(DATASET_DIR, split_name)
    print(f"\n--- {split_name} ---")

    total_images = 0
    corrupted_files = []
    sizes_seen = set()

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(image_files)
        total_images += count
        print(f"{class_name}: {count} images")

        # Check a sample of images in this class for size + corruption
        for filename in image_files:
            filepath = os.path.join(class_path, filename)
            try:
                with Image.open(filepath) as img:
                    sizes_seen.add(img.size)  # (width, height)
            except Exception as e:
                corrupted_files.append(filepath)

    print(f"Total in {split_name}: {total_images} images")
    print(f"Unique image sizes found: {len(sizes_seen)}")
    if len(sizes_seen) <= 5:
        print(f"Sizes: {sizes_seen}")
    else:
        print(f"Sample sizes: {list(sizes_seen)[:5]} ... (and {len(sizes_seen)-5} more)")

    if corrupted_files:
        print(f"WARNING: {len(corrupted_files)} corrupted/unreadable files found:")
        for f in corrupted_files[:10]:
            print(f"  {f}")
    else:
        print("No corrupted files found.")

inspect_split("Training")
inspect_split("Testing")
