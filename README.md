# Binary Classification of Glomeruli

This project develops a machine learning model to classify glomeruli images into sclerotic and non-sclerotic categories using PyTorch. The approach utilizes a pre-trained ResNet18 model and involves a series of steps from data preprocessing to model evaluation. The model is built using PyTorch and trained on a dataset provided by the CMIL at University of Florida. The dataset consists of glomeruli image patches labeled as sclerotic (1) or non-sclerotic (0).

## Approach

The project was approached with a systematic methodology, starting from understanding the dataset, preprocessing images, defining a suitable model architecture, training the model, and finally evaluating its performance.

## ML Pipeline:

The Machine Learning pipline includes the following steps:

1. __Library Imports and Setup:__ Initialization of necessary Python libraries and setting up paths for data storage and results.
2. __Data Reading and Description:__ Analysis of the image annotations contained within `public.csv` file to understand the distribution and characteristics of the data, which is crucial for guiding the data processing and the model training steps.
3. __Visualization:__

   __Category Distribution:__ Assessment of the balance between sclerotic and non-sclerotic samples by visualizing the category distribution for identifying any need for special handling during model training.
   
   __Image Size Distribution:__ Ensuring consistency in image dimensions across the dataset.
4. __Dataset Handling:__ Creation of a `GlomeruliDataset` class to manage data loading and preprocessing. This class, which extends `torch.utils.data.Dataset`, enables the application of image transformations and seamlessly integrates with PyTorch's data loading functionalities.
5. __Dataset Splitting:__ Segregation of the dataset into training, validation, and testing sets to facilitate model evaluation.
6. __Model Configuration:__ Adaptation of the pre-trained ResNet18 model for the binary classification task, and modification of  the final layer to suit the output requirements.
7. __Model Training:__ Execution of the training process, with progress tracked through the loss metric. Adam optimizer and CrossEntropyLoss are employed.
8. __Model Saving:__ Storage of the trained model on disk for subsequent evaluation.
9. __Model Evaluation:__ Evaluation of the model is performed on a seperate testing set, using accuracy, precision, recall, and F1 score as metrics.

## Data Pre-processing and Post-processing

# Preprocessing 
Includes resizing images to 224x224 pixels, converting them to tensor format, and normalizing based on predefined mean and standard deviation values.

# Post-processing:
Involves translating the model's output probabilities into categorical predictions (sclerotic or non-sclerotic).

## Dataset Division

The dataset is split into training, validation, and test sets with a proportion of 60%, 20%, and 20% respectively. This division ensures a balanced approach for model training, hyperparameter tuning, and evaluation. Data is loaded and batched using PyTorch's DataLoader utility, facilitating efficient and scalable model training.

## Performance Metrics

The model's performance is evaluated based on accuracy, precision, recall, and F1 score. These metrics are chosen to provide a comprehensive understanding of the model's ability to classify sclerotic and non-sclerotic glomeruli accurately.
- __Accuracy:__ 96.96%, indicating a high level of overall correct predictions.
- __Precision:__ 85.96%, showing the model's accuracy when predicting positive classes.
- __Recall:__ 98.49%, highlighting the model's ability to detect most positive instances.
- __F1 Score:__ 91.80%, balancing precision and recall, suggests a robust model performance across both classes.

## Environment Setup

Ensure you have Conda installed. Clone the project repository, then create and activate the project environment using:

   ```
   conda env create -f environment.yml
   conda activate project_env
   ```

## Model Training

Execute the following command to train the model:

   ```
   python main.py
   ```

## Model Evaluation
__Model download:__ [model](https://www.dropbox.com/scl/fo/ip7pxm8zgm4t23qes0e15/h?rlkey=23u1hed6lla5kbzcuc2e9muna&dl=0)

After training the model, you can evaluate it on a new set of images by using the `evaluation.py` script. The script requires the path to a folder containing glomeruli image patches as input and outputs a CSV file with the model's predictions.

Run the script as follows:

   ```
   python evaluation.py <path_to_image_folder> model/model.pth evaluation.csv
   ```

Replace `<path_to_image_folder>` with the path to your image folder.

## Dataset

The dataset is organized into two sub-folders within the `data/` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`, containing image patches for sclerotic and non-sclerotic classes, respectively. The `public.csv` file lists the image patches with their corresponding labels.

## Dependencies

This project is developed in PyCharm, under an Anaconda environment, and utilizes PyTorch for the deep learning components. Ensure you have Anaconda and PyCharm installed on your system. The following libraries and tools are required:

- __Python:__ Ensure you have Python installed, ideally through Anaconda. This project was developed using Python 3.10.
- __PyTorch:__ Used for all deep learning operations, including neural network definition and training. 
- __Pandas:__ For data manipulation and analysis. 
- __Pillow (PIL):__ For image processing tasks.
- __Matplotlib:__ For creating static, interactive, and animated visualizations in Python.
- __Seaborn:__ For making statistical graphics in Python. It is built on top of Matplotlib. 
- __Scikit-learn:__ For machine learning utilities such as data splitting. 
- __Warnings:__ Typically included with the standard Python library, used to suppress warnings.

For a complete list of dependencies, refer to the `environment.yml` file.

## Project Files
- `main.py`: Script for training the model.
- `evaluation.py`: Script for evaluating the model on a new set of glomeruli image patches.
- `data/`: Folder containing the dataset and `public.csv` with image names and their corresponding labels.
- `model/`: Directory where the trained model is saved.
- `environment.yml`: Conda environment file to set up the required environment for running the project.
- `test/`: Contains test data. 
- `evaluation.cvs`: Contains test results.
- `results/`: Folder containing the plots for data visualization and model evaluation.