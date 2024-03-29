# Binary Classification of Glomeruli

Glomeruli, crucial components of the kidney, play a pivotal role in renal function. Abnormalities in glomerular structure can serve as indicators of various renal diseases, making accurate classification of glomeruli images essential for diagnosis and treatment. 

In this project, I present a deep learning approach for glomeruli image classification using the ResNet18 architecture pretrained on the ImageNet dataset. I utilized data preprocessing techniques, including image resizing and normalization, to ensure uniformity and facilitate model training. Then adjusted the ResNet18 model to make to fit for the binary classification task. Through experimentation and evaluation, the algorithm can accurately distinguish between non-globally sclerotic and globally sclerotic glomeruli. The model achieves high accuracy, precision, and recall on a test set. Visualizations such as class distribution pie charts, image size box plots, and precision-recall curves provide rich information about the dataset characteristics and model performance. Overall, the experiment highlights the efficacy of pretrained deep learning models in glomeruli image classification, providing a promising usage for improving renal disease diagnosis and management.

## ML Pipeline:

The Machine Learning pipline includes the following steps:

1. __Library Imports and Setup:__ Essential Python libraries such as Pandas, PyTorch, and Matplotlib are imported. Setup involves configuring paths for data storage (`data/`) and results output (`results/`), and initializing hyperparameters like learning rate and batch size.

2. __Data Reading and Description:__ The annotations from the `public.csv` file are read using Pandas. This CSV file contains image annotations which are crucial for understanding the data distribution and guiding further processing for model training.

3. __Data Visualization:__

   __Category Distribution:__ The balance between sclerotic and non-sclerotic samples is visualized using a bar plot, addressing any potential class imbalance in the dataset.
   
   __Image Size Distribution:__ The distribution of image dimensions is assessed using histograms and box plots to ensure consistent image sizes are fed into the model.

4. __Dataset Handling and Preparation:__ Creation of a `GlomeruliDataset` class, inheriting from `torch.utils.data.Dataset`, is defined for efficient data loading and preprocessing. This includes image resizing, normalization, and tensor conversion. The dataset is split into training, validation, and testing sets.
5. __Model Configuration:__ The ResNet18 model, pre-trained on ImageNet, is adapted for the binary classification task. The last fully connected layer is modified to output two classes.
6. __Model Training:__ The training process is carried out over a specified number of epochs. During training, model parameters are updated using the Adam optimizer, and learning rate adjustments are made with a StepLR scheduler. Training and validation losses and accuracies are tracked for performance monitoring.
7. __Model Saving:__ Post-training, the model's state dictionary is saved to disk for future evaluation and deployment, ensuring that the trained model can be reused without the need for retraining.
8. __Model Evaluation:__ The model is evaluated on a separate testing set using metrics like accuracy, precision, recall, and F1 score. Additional performance insights are provided through a confusion matrix, precision-recall curve, and ROC curve.

## Dataset and Preprocessing
The dataset is organized into two sub-folders within the `data/` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`, containing image patches for sclerotic and non-sclerotic classes, respectively. The `public.csv` file lists the image patches with their corresponding labels.

The dataset consists of 5758 histopathological images labeled as globally sclerotic (1) or non-globally sclerotic (0). Each image underwent preprocessing, including resizing, normalization, and augmentation, to ensure consistent input to the neural network.

## Training and Validation
ResNet18 as a residual learning framework for deep convolutional neural networks, is selected for this task due to its efficiency in learning from a relatively small amount of data and its ability to generalize well on image recognition tasks. Pretrained on ImageNet, it has proven to be an effective architecture for transfer learning, especially in medical imaging.

I adopted the ResNet18 model, modifying the final layer to output binary classifications. The dataset was split into training (60%), validation (20%), and test sets (20%). 

## Performance Metrics

The model's performance is evaluated based on accuracy, precision, recall, and F1 score. These metrics are chosen to provide a comprehensive understanding of the model's ability to classify sclerotic and non-sclerotic glomeruli accurately.
- __Accuracy:__ 98.70%, indicating a high level of overall correct predictions.
- __Precision:__ 95.10%, showing the model's accuracy when predicting positive classes.
- __Recall:__ 97.49%, highlighting the model's ability to detect most positive instances.
- __F1 Score:__ 96.28%, balancing precision and recall, suggests a robust model performance across both classes.

## Conclusion

The high performance of the [model](https://www.dropbox.com/scl/fo/ip7pxm8zgm4t23qes0e15/h?rlkey=23u1hed6lla5kbzcuc2e9muna&dl=0) suggests that transfer learning and fine-tuning of pre-trained networks are effective for binary classification in histopathological images.

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
- `train.py`: Script for training the model.
- `evaluation.py`: Script for evaluating the model on a new set of glomeruli image patches.
- `data/`: Folder containing the dataset and `public.csv` with image names and their corresponding labels.
- `model/`: Directory where the trained model is saved.
- `environment.yml`: Conda environment file to set up the required environment for running the project.
- `test/`: Contains test data. 
- `evaluation.cvs`: Contains test results.
- `results/`: Folder containing the plots for data visualization and model evaluation.

# Reference
Sadia Showkat, Shaima Qureshi,
Efficacy of Transfer Learning-based ResNet models in Chest X-ray image classification for detecting COVID-19 Pneumonia,
Chemometrics and Intelligent Laboratory Systems,
Volume 224,
2022,
104534,
ISSN 0169-7439,
https://doi.org/10.1016/j.chemolab.2022.104534.