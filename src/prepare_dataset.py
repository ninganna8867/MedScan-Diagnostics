import random
import shutil
from pathlib import Path

random.seed(42)

source = Path("data/raw/COVID-19_Radiography_Dataset")
destination = Path("data/processed")

classes = ["COVID", "Normal", "Viral Pneumonia"]
images_per_class = 1000

for class_name in classes:
    source_folder = source / class_name / "images"
    destination_folder = destination / class_name

    images = sorted(source_folder.glob("*.png"))

    if len(images) < images_per_class:
        raise ValueError(f"Not enough images in {class_name}")

    selected_images = random.sample(images, images_per_class)

    for image_path in selected_images:
        shutil.copy2(image_path, destination_folder / image_path.name)

    print(f"{class_name}: {len(selected_images)} images copied")

print("\nBalanced dataset preparation completed!")
