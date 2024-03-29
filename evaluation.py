import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# Define a class that extends PyTorch's Dataset class. This custom dataset
# is designed to handle loading images from a directory for use with a neural network.
# It includes functionality to process and transform the images as required for model input.
class ImageDataset(Dataset):
    # Initialize the dataset with the directory of images and any transformations.
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Directory where images are located.
        self.transform = transform  # Transformations to be applied to each image.
        self.images = os.listdir(root_dir)  # List all files in the directory.

    # Return the number of images in the dataset.
    def __len__(self):
        return len(self.images)

    # Retrieve an image by index, apply transformations, and return it.
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])  # Get image path.
        image = Image.open(img_name).convert('RGB')  # Open and convert to RGB for consistency.
        if self.transform:
            image = self.transform(image)  # Apply the predefined transformations.
        return image, self.images[idx]  # Return the transformed image and its filename.


# Load a pre-trained ResNet18 model from a saved file.
# The ResNet18 is a model that has proven effective in image recognition tasks.
def load_model(model_path):
    model = models.resnet18()  # Instantiate the ResNet18 model.
    # Adjust the fully connected layer for binary classification
    num_ftrs = model.fc.in_features  # Get the input feature size for the final layer.
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Replace the final layer for binary classification.
    model.load_state_dict(torch.load(model_path))  # Load the saved model weights.
    model.eval()  # Set the model to evaluation mode.
    return model


# Define a function to perform predictions using the model and a dataloader.
# This function will feed data through the model and collect the predictions.
def predict(model, dataloader, device):
    predictions = []  # Store predictions.
    files = []  # Keep track of file names.
    with torch.no_grad():  # Disable gradient computation for efficiency.
        for images, image_files in dataloader:  # Iterate over batches of images.
            images = images.to(device)  # Move images to the device (GPU or CPU).
            outputs = model(images)  # Get model outputs.
            _, predicted = torch.max(outputs.data, 1)  # Find the index with the highest score.
            predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU and convert to numpy.
            files.extend(image_files)  # Save the file names.
    return files, predictions


# The main function that ties everything together for prediction.
def main(folder_path, model_path, output_csv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Determine the computing device.

    # Define the transformations to be applied to each image.
    # These should match the transformations applied during model training.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the size the model expects.
        transforms.ToTensor(),  # Convert images to tensor format.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images.
    ])

    # Create an instance of the dataset with the given folder path and transformations.
    dataset = ImageDataset(root_dir=folder_path, transform=transform)
    # DataLoader allows for batch processing of images, and shuffling is set to False for prediction.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the model and send it to the device.
    model = load_model(model_path).to(device)

    # Predict the labels of the images using the model and dataloader.
    files, predictions = predict(model, dataloader, device)

    # Create a DataFrame with the filenames and their corresponding predicted labels.
    # This is useful for analysis and reviewing model performance.
    df = pd.DataFrame({
        'Image': files,
        'Label': predictions
    })
    # Save the DataFrame to a CSV file, which can be shared or used for further processing.
    df.to_csv(output_csv, index=False)
    print(f"Prediction results saved to {output_csv}")


# Check if the script is being run as the main program and not as a module.
if __name__ == "__main__":
    # If this script is executed from the command line, sys.argv will contain the command line arguments.
    # sys.argv[1] is expected to be the folder path where images for prediction are stored.
    # sys.argv[2] is expected to be the path to the trained model file.
    # sys.argv[3] is expected to be the path to the output CSV file where predictions should be saved.
    # These command line arguments allow for flexibility and ease of use from the command line interface.
    folder_path = sys.argv[1]
    model_path = sys.argv[2]
    output_csv = sys.argv[3]
    # Call the main function with the provided arguments, initiating the prediction process.
    # This is the entry point for the script when running from the command line.
    main(folder_path, model_path, output_csv)
