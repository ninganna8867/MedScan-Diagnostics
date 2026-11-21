
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from data_loader import test_loader, train_dataset
from model import create_model

# Select CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the trained ResNet-18 model
model = create_model(num_classes=3)

model_path = "models/best_resnet18.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Get class names
class_names = train_dataset.classes
print("Classes:", class_names)

# Store actual labels and predictions
all_labels = []
all_predictions = []
all_probabilities = []

# Evaluate using the independent test dataset
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Print classification report
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)

# Print confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("\nConfusion Matrix:")
print(cm)

# Calculate multiclass ROC-AUC
binary_labels = label_binarize(
    all_labels,
    classes=list(range(len(class_names)))
)

auc = roc_auc_score(
    binary_labels,
    all_probabilities,
    average="macro",
    multi_class="ovr"
)

print("\nMacro ROC-AUC:", round(auc, 4))

# Calculate COVID false-negative rate
covid_index = class_names.index("COVID")

covid_true_positive = cm[covid_index, covid_index]
covid_false_negative = cm[covid_index, :].sum() - covid_true_positive

covid_fn_rate = (
    covid_false_negative /
    (covid_true_positive + covid_false_negative)
)

print(
    "\nCOVID False Negative Rate:",
    round(covid_fn_rate * 100, 2),
    "%"
)

print("\nIndependent test evaluation completed!")
