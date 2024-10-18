# Amazon Product Reviews Sentiment Analysis

This project implements a sentiment analysis model for Amazon product reviews using a fine-tuned DistilBERT model with LoRA (Low-Rank Adaptation). The model classifies reviews into three categories: positive, neutral, and negative.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [License](#license)

## Features

- Load and preprocess data from a CSV file.
- Balance the dataset using resampling techniques.
- Train a DistilBERT model with cross-validation.
- Evaluate model performance using metrics like accuracy, precision, recall, and F1 score.
- Generate confusion matrices for visualizing predictions.

## Requirements

Make sure you have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `peft`

You can install the necessary packages using pip:

```bash
pip install numpy pandas matplotlib seaborn torch transformers datasets scikit-learn peft
```
## Installation
Clone this repository:
```
git clone <repository-url>
cd <repository-folder>
```
Install the required packages as mentioned above.

Prepare your dataset and ensure it is in the correct format.

## Usage
Data Preparation
Place your dataset CSV file in the specified path (FILE_PATH in the code).
Modify the COLUMNS_TO_KEEP constant if your dataset has different column names.

## Training
To train the model, run the following command:

```
python train.py
```

This will load the dataset, balance it, and start the training process. The trained model will be saved in the ./saved_model directory.

## Testing
To test the model, you can use the following command:

```

python test.py
```

Make sure to set the TEST_FILE_PATH variable to point to your test data CSV file.

## Results
The training script will output average metrics (accuracy, precision, recall, F1 score) and display a confusion matrix of the predictions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```

Make sure to replace `<repository-url>` and `<repository-folder>` with the actual URL of your repository and the folder name, respectively. You can also adjust any sections to better match the specifics of your project or your coding style!
```

