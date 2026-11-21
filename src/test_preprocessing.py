from pathlib import Path
from PIL import Image
from preprocessing import test_transform

image_path = Path("data/processed/train/COVID")

image_file = next(image_path.glob("*.png"))

image = Image.open(image_file).convert("RGB")

processed_image = test_transform(image)

print("Original image size:", image.size)
print("Processed image shape:", processed_image.shape)
print("Tensor data type:", processed_image.dtype)
print("Preprocessing test successful!")
