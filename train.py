# Glomeruli Image Classification with Pre-trained ResNet18
# 1. Set Up Environment
# Import the necessary libraries required throughout the program.
# os for file path operations, pandas for data handling, PIL for image processing,
# matplotlib and seaborn for visualization, sklearn for machine learning utilities,
# torchvision for computer vision tasks, torch for deep learning model operations,
# and warnings to handle future warnings that may not be relevant to current execution.
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torchvision import models

# Suppress future warnings from libraries to maintain clarity in output.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Exploration and Pipeline Setup
# Set the paths for the data directory and the results directory where outputs will be saved.
base_dir = 'data'
results_dir = 'results'  # Directory to save the output visualizations.
csv_file = os.path.join(base_dir, 'public.csv')  # Path to CSV file containing image annotations
root_dir = base_dir  # Root directory where image data is stored
learning_rate = 0.001  # Learning rate for the optimizer, affects how much we update model weights during training
batch_size = 32  # Size of image batches to be processed by the model, affects memory usage and speed
num_epochs = 15  # Number of complete passes through the dataset for training

# Ensure the results directory exists to save visualization outputs, create it if it doesn't.
os.makedirs(results_dir, exist_ok=True)

# Try to load the annotations CSV and handle any file reading errors gracefully.
try:
    annotations = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Read the CSV file containing image annotations using pandas.
# This CSV contains important data that will be used for training the model.
annotations = pd.read_csv(csv_file)

# After successful loading, print basic statistics to understand data distribution.
print(annotations.describe())
"""
       ground truth
count   5758.000000
mean       0.183050
std        0.386741
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        1.000000
"""

# Category distribution visualization to understand class balance which is important for model training.
print("\nCategory distribution:")
print(annotations['ground truth'].value_counts())
"""
       ground truth
count   5758.000000
mean       0.183050
std        0.386741
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        1.000000
"""

# Using seaborn, create a bar plot for a clear visual of category counts.
# This may reveal class imbalances that could necessitate techniques like weighted loss during training.
plt.figure(figsize=(6, 4))
sns.countplot(x='ground truth', data=annotations)
plt.title('Category distribution')
plt.xlabel('Category')
plt.ylabel('Count')
category_distribution_fig = os.path.join(results_dir, 'category_distribution.png')
plt.savefig(category_distribution_fig)
plt.close()
"""
Category distribution:
ground truth
0    4704
1    1054
Name: count, dtype: int64
"""

# Image dimensions collection for ensuring consistent model input.
img_dims = {'height': [], 'width': []}
for idx in range(len(annotations)):
    # Determine the subdirectory based on the category of the image.
    img_label = annotations.iloc[idx, 1]
    subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
    img_path = os.path.join(root_dir, subdir, annotations.iloc[idx, 0])
    with Image.open(img_path) as img:
        # Record the dimensions of each image.
        img_dims['height'].append(img.height)
        img_dims['width'].append(img.width)

# Visualize and save the distribution of image sizes using matplotlib.
# It is important for the preprocessing steps to know these dimensions ahead.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(img_dims['height'], bins=20, kde=True)
plt.title('Image height distribution')
plt.xlabel('Height')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(img_dims['width'], bins=20, kde=True)
plt.title('Image width distribution')
plt.xlabel('Width')
plt.ylabel('Count')
plt.tight_layout()
size_distribution_fig = os.path.join(results_dir, 'size_distribution.png')
plt.savefig(size_distribution_fig)
plt.close()


# Dataset Preparation
# Custom dataset class for handling the glomeruli images.
# Inherits from PyTorch's Dataset class for compatibility with DataLoader and other PyTorch utilities.
class GlomeruliDataset(Dataset):
    def __init__(self, annotations_df, root_dir, transform=None):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_label = self.annotations.iloc[idx, 1]
        subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
        img_name = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found. Skipping.")
            return None  # Skip missing files

        if self.transform:
            image = self.transform(image)
        return image, int(img_label)


# Define transformations for image preprocessing.
# Includes resizing to the expected input size, conversion to tensor, and normalization using ImageNet's mean and std.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the dataset with the CSV annotations and the defined transformations.
dataset = GlomeruliDataset(annotations_df=annotations, root_dir=root_dir, transform=transform)

# Split the dataset into train, validation, and test sets using sklearn's train_test_split.
# This ensures an unbiased evaluation of the model's performance.
train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Further split the training+validation set into distinct training and validation sets.
train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Prepare DataLoader instances for the training, validation, and test sets.
# DataLoader provides an iterable over the dataset with additional features like shuffling and batching.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print("Dataset preparation complete.")

# Model Preparation
# Configure the pre-trained ResNet18 model for binary classification by replacing the final layer.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjust for the binary classification task

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer selection.
# CrossEntropyLoss is suited for classification tasks, and Adam is an effective optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Learning rate scheduler

# Training and Validation
# Training loop involving forward and backward passes, optimizer steps, and scheduler stepping.
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Epochs represent complete passes over the dataset. In each epoch, the model learns from the training set and is validated.
for epoch in range(num_epochs):
    # Set the model to training mode. This is crucial because certain layers like BatchNorm and Dropout
    # behave differently during training compared to during testing.
    model.train()

    # Initialize variables to track training progress within each epoch.
    train_loss, train_correct, train_total = 0, 0, 0

    # Iterate over the training dataset. DataLoader provides batches of the dataset.
    for images, labels in train_loader:
        # Move images and labels to the device (GPU or CPU). This step is necessary because the model is also on the
        # same device.
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients. This is necessary as gradients accumulate by default, for every backward pass.
        optimizer.zero_grad()

        # Forward pass: Compute predicted outputs by passing inputs to the model.
        outputs = model(images)

        # Compute the loss based on model output and actual labels.
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters.
        loss.backward()

        # Perform a single optimization step (parameter update).
        optimizer.step()

        # Update training loss and accuracy calculation.
        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    # Validation phase
    # Switch model to evaluation mode to turn off layers like BatchNorm and Dropout.
    model.eval()

    # Initialize variables to track validation progress.
    val_loss, val_correct, val_total = 0, 0, 0

    # Disable gradient calculation to save memory and computations since it's not needed for evaluation.
    with torch.no_grad():
        for images, labels in val_loader:
            # Similar to the training loop, but no need to zero gradients or perform backward pass.
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update validation loss and accuracy calculation.
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

        # Step the scheduler
        scheduler.step()

        # Calculate average losses and accuracy
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # Append to lists for logging and plotting
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

    # Plotting and saving training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    accuracy_plot_path = os.path.join(results_dir, 'training_validation_accuracy.png')
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"Plots saved to {results_dir}")

# Save the trained model to disk for later use or further evaluation.
model_save_path = 'model/model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists.
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate model performance on unseen data to assess generalization.model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
"""
Accuracy: 0.9870
Precision: 0.9510
Recall: 0.9749
F1 Score: 0.9628
"""

# Plot a pie chart to visualize the distribution of classes in the dataset.
class_distribution = annotations['ground truth'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.savefig(os.path.join(results_dir, 'class_distribution_pie_chart.png'))
plt.close()

# Plot a box plot to visualize the distribution of image sizes (height and width).
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(img_dims))
plt.title('Image Size Distribution')
plt.ylabel('Pixels')
plt.savefig(os.path.join(results_dir, 'image_size_boxplot.png'))
plt.close()

# Plot the training and validation loss across epochs.
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(results_dir, 'training_validation_loss.png'))
plt.close()

# Plot the training and validation accuracy across epochs.
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'training_validation_accuracy.png'))
plt.close()

# Plot the confusion matrix to visualize model performance.
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Plot the precision-recall curve to evaluate model performance.
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
plt.close()

# Plot the ROC curve to evaluate model performance.
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
plt.close()
