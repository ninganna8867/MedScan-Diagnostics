from data_loader import train_loader

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Image data type:", images.dtype)
print("Labels:", labels)
print("First batch loaded successfully!")
