# Binary Classification of Glomeruli

This project aims to develop a machine learning model capable of distinguishing between sclerotic and non-sclerotic glomeruli using image data. The model is built using PyTorch and trained on a dataset provided by a large medical consortium. The dataset consists of glomeruli image patches labeled as sclerotic (1) or non-sclerotic (0).

## Project Structure

- `main.ipynb`: Jupyter notebook containing the code for training the model.
- `evaluation.py`: Script for evaluating the model on a new set of glomeruli image patches.
- `Data/`: Folder containing the dataset and `public.csv` with image names and their corresponding labels.
- `model/`: Directory where the trained model is saved.
- `environment.yml`: Conda environment file to set up the required environment for running the project.
- `test`: Test data 

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

The dataset is organized into two sub-folders within the `Data/` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`, containing image patches for sclerotic and non-sclerotic classes, respectively. The `public.csv` file lists the image patches with their corresponding labels.

## Model Training

To train the model, open the `main.ipynb` notebook in Jupyter or JupyterLab:

```
jupyter notebook main.ipynb
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

- Python 3.11
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
