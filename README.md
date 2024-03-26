# Binary Classification of Glomeruli

This project aims to develop a machine learning model capable of distinguishing between sclerotic and non-sclerotic glomeruli using image data. The model is built using PyTorch and trained on a dataset provided by a large medical consortium. The dataset consists of glomeruli image patches labeled as sclerotic (1) or non-sclerotic (0).

## Project Structure

- `main.py`: Script for training the model.
- `evaluation.py`: Script for evaluating the model on a new set of glomeruli image patches.
- `data/`: Folder containing the dataset and `public.csv` with image names and their corresponding labels.
- `model/`: Directory where the trained model is saved.
- `environment.yml`: Conda environment file to set up the required environment for running the project.
- `test/`: Test data 

## The ML pipeline includes the following steps:

1. Importing necessary libraries and setting up data paths.
2. Reading and describing the image annotations from a CSV file.
3. Visualizing the distribution of categories in the dataset.
4. Checking and visualizing the size distribution of images.
5. Defining a `GlomeruliDataset` class for efficient data handling and image transformations.
6. Splitting the dataset into training, validation, and testing sets.
7. Setting up and training a pre-trained ResNet18 model, adjusting the final layer to match the binary classification task.
8. Training the model on the dataset and printing the loss after each epoch.
9. Saving the trained model to disk.
10. Evaluating the model on the test set and printing the accuracy.

## Pre-processing and Post-processing

Pre-processing involves resizing the images to 224x224 pixels, normalizing them with specific mean and standard deviation values, and converting them into tensor format for processing by the neural network.

Post-processing involves mapping the output tensor from the neural network to a binary class prediction.

## Dataset Division

The dataset is divided as follows:
- Training set: 80%
- Validation set: Not explicitly created but can be set up similarly.
- Test set: 20%

The division is performed using a random split with a fixed seed to ensure reproducibility.

## Performance Metrics

The model's performance is evaluated using accuracy, the proportion of correctly classified images over the total number of images in the test set.

## Environment Setup

Ensure you have Conda installed on your system. To set up the project environment:

1. Clone the repository to your local machine.
2. Navigate to the project directory in your terminal.
3. Create the Conda environment using the `environment.yml` file:

   ```
   conda env create -f environment.yml
   ```

4. Activate the environment:

   ```
   conda activate project_env
   ```

## Dataset

The dataset is organized into two sub-folders within the `data/` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`, containing image patches for sclerotic and non-sclerotic classes, respectively. The `public.csv` file lists the image patches with their corresponding labels.

## Model Training

To train the model, open the `main.py` in pyCharm:

```
python main.py
```

Follow the instructions and code cells within the notebook to train the model. The notebook includes data preprocessing, model definition, training, and validation phases.

## Model Evaluation
Model download: [model](https://www.dropbox.com/scl/fo/ip7pxm8zgm4t23qes0e15/h?rlkey=23u1hed6lla5kbzcuc2e9muna&dl=0)

After training the model, you can evaluate it on a new set of images by using the `evaluation.py` script. The script requires the path to a folder containing glomeruli image patches as input and outputs a CSV file with the model's predictions.

Run the script as follows:

```
python evaluation.py <path_to_image_folder> model/model.pth evaluation.csv
```

Replace `<path_to_image_folder>` with the path to your image folder.

## Dependencies

- Python 3.10
- PyTorch
- torchvision 0.17.1
- PIL
- pandas
- numpy
- scikit-learn
- matplotlib

For a complete list of dependencies, refer to the `environment.yml` file.

## Performance Metrics

The model's performance is evaluated based on accuracy, precision, recall, and F1 score. These metrics are chosen to provide a comprehensive understanding of the model's ability to classify sclerotic and non-sclerotic glomeruli accurately.
