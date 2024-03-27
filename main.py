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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torchvision import models

# Suppress future warnings from libraries to maintain clarity in output.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Exploration
# Hyperparameters and paths setup
base_dir = 'data'
results_dir = 'results'  # Where to save the output visualizations.
csv_file = os.path.join(base_dir, 'public.csv')
root_dir = base_dir
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Ensure that the results directory exists; create it if it does not.
os.makedirs(results_dir, exist_ok=True)

# Error handling for file reading
try:
    annotations = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Read the CSV file containing image annotations using pandas.
# This CSV contains important data that will be used for training the model.
annotations = pd.read_csv(csv_file)
# Print out the basic statistics of the CSV file to understand the data distribution.
print(annotations.describe())

# 3. Data Visualization
# Display the distribution of the categories in the dataset.
# This is crucial to understand the balance of classes we're dealing with.
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

# Visualize and save the category distribution using seaborn for a clearer picture.
# This can highlight class imbalances that may require special handling during model training.
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

# Check the distribution of image sizes to ensure that the model receives images of consistent size.
# Differing image sizes may affect model performance and training dynamics.
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

# 4. Dataset Preparation
# Define the dataset class for the Glomeruli images.
# This class inherits from torch.utils.data.Dataset and is tailored for our specific data structure.
# It allows for easy integration with PyTorch's data loading utilities later on.
class GlomeruliDataset(Dataset):
    # Initialization method to set up the necessary variables.
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # Method to get the number of items in the dataset.
    def __len__(self):
        return len(self.annotations)

    # Method to retrieve a specific item from the dataset by index.
    def __getitem__(self, idx):
        img_label = self.annotations.iloc[idx, 1]
        subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
        img_name = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Image {img_name} not found.")
            return None

        if self.transform:
            # If a transform is specified, apply it to the image. This usually includes resizing, normalizing,
            # and converting to a tensor.
            image = self.transform(image)
        return image, int(img_label)  # Return the transformed image and its label as a tuple.


# Preprocessing transformations are crucial for model training, ensuring all images are of the same size and scale.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # # Resize all images to the size expected by the model.
    transforms.ToTensor(),  # Convert images to PyTorch tensors.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the dataset, providing the path to the annotations CSV, the root directory of the images,
# and the transform.
dataset = GlomeruliDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Split the dataset into training and test sets for model evaluation.
# This is essential to assess model performance on unseen data.
train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Further split the training+validation set into distinct training and validation sets.
train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Prepare DataLoader instances for the training, validation, and test sets.
# DataLoader provides an iterable over the dataset with additional features like shuffling and batching.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print("Dataset preparation complete.")

# 5. Model Preparation
# Model setup involves using a pre-trained ResNet18 model.
# Pre-trained models use weights that have been learned on a large benchmark dataset,
# which can lead to faster convergence and improved performance.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features  # Get the number of input features to the fully connected layer.
model.fc = nn.Linear(num_ftrs, 2)  # Adjust the fully connected layer to match our number of target classes.

# Set the device to GPU if available for faster computation, otherwise use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer for the training process.
# CrossEntropyLoss is commonly used for classification tasks.
# Adam optimizer is chosen for its adaptiveness in learning rates.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training and Validation
# Train and validate the model for a specified number of epochs.
# Epochs are complete passes over the entire dataset.
# Training involves forward passes to compute the loss, and backward passes to update the weights.

# Initialize lists for logging metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

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
            # Similar to the training loop, but we don't need to zero gradients or perform backward pass.
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update validation loss and accuracy calculation.
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

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

        # Print training and validation results
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

    # Plotting and saving training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Specify the path and filename for the loss plot
    loss_plot_path = os.path.join(results_dir, 'training_validation_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()  # Close the plot to free up memory

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

# 7. Model Saving
# Save the trained model to disk for later use or further evaluation.
model_save_path = 'model/model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists.
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 8. Evaluation
# Evaluate the model on the test set.
# Evaluation is done in no_grad mode, which disables gradient computation,
# making it more memory-efficient and faster for evaluation.
model.eval()
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
Accuracy: 0.9861
Precision: 0.9378
Recall: 0.9849
F1 Score: 0.9608
"""