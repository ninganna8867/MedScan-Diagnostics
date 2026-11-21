import random
import shutil
from pathlib import Path

random.seed(42)

source = Path("data/processed")
classes = ["COVID", "Normal", "Viral Pneumonia"]

splits = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

for class_name in classes:
    images = sorted((source / class_name).glob("*.png"))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    split_images = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, selected_images in split_images.items():
        destination = source / split_name / class_name

        for image in selected_images:
            shutil.copy2(image, destination / image.name)

        print(f"{split_name} - {class_name}: {len(selected_images)} images")

print("\nDataset splitting completed!")
