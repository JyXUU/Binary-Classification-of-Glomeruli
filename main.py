# 1. Imports necessary libraries and sets up paths for data.
# 2. Reads and describes a CSV file containing image annotations.
# 3. Visualizes the distribution of categories within the dataset.
# 4. Checks and visualizes the size distribution of images.
# 5. Defines a `GlomeruliDataset` class for handling the dataset, including image transformations.
# 6. Splits the dataset into training and testing sets and prepares DataLoader instances for both.
# 7. Sets up a pre-trained ResNet18 model, adjusting the final layer to match the number of target classes.
# 8. Trains the model on the dataset, printing the loss after each epoch.
# 9. Saves the trained model to disk.
# 10. Evaluates the model on the test set, printing the accuracy.

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torchvision import models

# Set base path
base_dir = 'data'
results_dir = 'results'  # Directory to save results
csv_file = os.path.join(base_dir, 'public.csv')
root_dir = base_dir

# Read CSV file
annotations = pd.read_csv(csv_file)
print(annotations.describe())

# Category distribution
print("\nCategory distribution:")
print(annotations['ground truth'].value_counts())

# Visualize Category distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='ground truth', data=annotations)
plt.title('Category distribution')
plt.xlabel('Category')
plt.ylabel('Count')
category_distribution_fig = os.path.join(results_dir, 'category_distribution.png')
plt.savefig(category_distribution_fig)

# Check image size distribution
img_dims = {'height': [], 'width': []}
for idx in range(len(annotations)):
    img_label = annotations.iloc[idx, 1]
    if img_label == 0:
        subdir = "non_globally_sclerotic_glomeruli"
    else:
        subdir = "globally_sclerotic_glomeruli"
    img_path = os.path.join(root_dir, subdir, annotations.iloc[idx, 0])
    with Image.open(img_path) as img:
        img_dims['height'].append(img.height)
        img_dims['width'].append(img.width)

# Visualize image size distribution
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


class GlomeruliDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_label = self.annotations.iloc[idx, 1]
        subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
        img_name = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(img_label)

        if self.transform:
            image = self.transform(image)
        return (image, label)


# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = GlomeruliDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Split dataset into training and test sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print("Dataset preparation complete.")

# Model setup
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

print("Training complete.")
model_save_path = 'model/model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Model Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Model accuracy on the test set: {100 * correct / total}%')
