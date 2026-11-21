
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from model import create_model
from data_loader import test_loader, train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(num_classes=3)
model.load_state_dict(
    torch.load("models/experiment_config2.pth", map_location=device)
)
model.to(device)
model.eval()

actual = []
predicted = []
probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        actual.extend(labels.numpy())
        predicted.extend(outputs.argmax(dim=1).cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

accuracy = accuracy_score(actual, predicted)
auc = roc_auc_score(
    actual,
    probabilities,
    multi_class="ovr",
    average="macro"
)

print("\nCONFIGURATION 2 TEST RESULTS")
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("Macro ROC-AUC:", round(auc, 4))

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=train_dataset.classes
))

print("Confusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nEvaluation completed!")
