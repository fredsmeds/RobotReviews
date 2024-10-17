import pandas as pd
import torch
import matplotlib.pyplot as plt
import re
from transformers import RobertaTokenizer
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.utils import resample
from sklearn.model_selection import KFold
import numpy as np
import os

os.environ["WANDB_MODE"] = "disabled"

# Load the dataset
file_path = "/notebooks/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"  # Replace with the correct file path
df = pd.read_csv(file_path)

# Select the specified columns
columns_to_keep = [
    "name",
    "brand",
    "primaryCategories",
    "reviews.text",
    "reviews.rating",
]
df_selected = df[columns_to_keep]

# Show the first few rows to verify
df_selected.head()

# Plotting to visualize data balances


# Set plot size and style
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

# Plot the distribution of ratings
sns.countplot(x="reviews.rating", data=df_selected, palette="viridis")

# Set title and labels
plt.title("Distribution of Review Ratings", fontsize=16)
plt.xlabel("Review Ratings", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Show plot
plt.show()


# Separate the dataset by rating categories (assume 1-2 is negative, 3 is neutral, 4-5 is positive)
positive_reviews = df_selected[df_selected["reviews.rating"] >= 4]
neutral_reviews = df_selected[df_selected["reviews.rating"] == 3]
negative_reviews = df_selected[df_selected["reviews.rating"] <= 2]

# Find the class with the most samples
max_class_size = max(len(positive_reviews), len(neutral_reviews), len(negative_reviews))

# Oversample the minority classes to match the largest class
positive_upsampled = resample(
    positive_reviews, replace=True, n_samples=max_class_size, random_state=42
)
neutral_upsampled = resample(
    neutral_reviews, replace=True, n_samples=max_class_size, random_state=42
)
negative_upsampled = resample(
    negative_reviews, replace=True, n_samples=max_class_size, random_state=42
)

# Combine the upsampled datasets
df_balanced = pd.concat([positive_upsampled, neutral_upsampled, negative_upsampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# After balancing the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Map the labels to a new range
df_balanced["reviews.rating"] = df_balanced["reviews.rating"].map(
    {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
)


# Verify class balance
print(df_balanced["reviews.rating"].value_counts())


def map_to_sentiment(rating):
    if rating <= 2:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:
        return 2  # positive


df_balanced["labels"] = df_balanced["reviews.rating"].apply(map_to_sentiment)

from sklearn.model_selection import train_test_split

# Split the dataset before tokenization (TO AVOID DATA LEAKAGE)
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["reviews.text"],
    df_balanced["reviews.rating"],
    test_size=0.2,
    random_state=42,
)

# Ensure that X_train and y_train are Pandas DataFrames or Series before the loop
print(type(X_train))
print(type(y_train))

from transformers import DistilBertTokenizer

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize training and testing sets with padding and truncation
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)
test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)

# Print the shapes to verify consistency
print(f"Train encodings shape: {train_encodings['input_ids'].shape}")
print(f"Test encodings shape: {test_encodings['input_ids'].shape}")

# Check attention mask shapes
print(f"Train attention mask shape: {train_encodings['attention_mask'].shape}")
print(f"Test attention mask shape: {test_encodings['attention_mask'].shape}")

from datasets import Dataset

# Create HuggingFace Datasets for training and test data
train_dataset = Dataset.from_dict(
    {
        "input_ids": train_encodings["input_ids"].tolist(),
        "attention_mask": train_encodings["attention_mask"].tolist(),
        "labels": y_train.tolist(),
    }
)

test_dataset = Dataset.from_dict(
    {
        "input_ids": test_encodings["input_ids"].tolist(),
        "attention_mask": test_encodings["attention_mask"].tolist(),
        "labels": y_test.tolist(),
    }
)

# Verify the shape of the dataset to ensure proper tokenization
print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import torch

# Assuming you have already tokenized the data correctly as seen in the previous images
# Tokenization (already done by you)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

from peft import LoraConfig, TaskType, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    r=8,  # Low-rank adaptation dimension
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout in LoRA layers
    target_modules=[
        "q_lin",
        "v_lin",
    ],  # Target modules for LoRA in DistilBERT (MultiHeadAttention)
)

# Load pretrained model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,  # Adjust based on your labels (e.g. positive, negative, neutral)
)

# Wrap the model with LoRA using PEFT
model = get_peft_model(model, lora_config)

from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Adjust based on your need
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    no_cuda=False,  # Assuming you have a GPU, if not set to True
)
# Create HuggingFace Datasets for training and testing data (already done by you)
train_dataset = Dataset.from_dict(
    {
        "input_ids": train_encodings["input_ids"].tolist(),
        "attention_mask": train_encodings["attention_mask"].tolist(),
        "labels": y_train.tolist(),
    }
)

test_dataset = Dataset.from_dict(
    {
        "input_ids": test_encodings["input_ids"].tolist(),
        "attention_mask": test_encodings["attention_mask"].tolist(),
        "labels": y_test.tolist(),
    }
)

# Define the Trainer instance
trainer = Trainer(
    model=model,  # PEFT-wrapped model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model with LoRA
trainer.train()


def plot_metrics(trainer):
    # Convert log history to a DataFrame for easier plotting
    log_history = pd.DataFrame(trainer.state.log_history)

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        log_history["step"], log_history["loss"], label="Training Loss", color="blue"
    )
    plt.title("Training Loss Over Time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # If there are evaluation logs, plot validation loss and other metrics
    if "eval_loss" in log_history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(
            log_history["step"],
            log_history["eval_loss"],
            label="Validation Loss",
            color="red",
        )
        plt.title("Validation Loss Over Time")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "eval_accuracy" in log_history.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(
            log_history["step"],
            log_history["eval_accuracy"],
            label="Validation Accuracy",
            color="green",
        )
        plt.title("Validation Accuracy Over Time")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()


print(trainer.state.log_history)
print(X_train.shape)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np

# Define the number of splits (K in K-fold)
k_folds = 2
kf = KFold(n_splits=k_folds)

# Create empty lists to store the metrics for each fold
fold_metrics = []

X_train = X_train.iloc[:100]  # Limit to first 100 rows
y_train = y_train.iloc[:100]


# Assuming X_train and y_train are your features and labels (from the previous snippets)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")

    # Split the training and validation sets
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Load the model and fine-tune it on the fold training data (using LoRA as standard)
    trainer.train()

    # Generate predictions for the validation set
    predictions = trainer.predict(X_fold_val)

    # Convert predictions to labels (assumed argmax over logits)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Calculate metrics
    accuracy = accuracy_score(y_fold_val, preds)
    precision = precision_score(y_fold_val, preds, average="weighted")
    recall = recall_score(y_fold_val, preds, average="weighted")
    f1 = f1_score(y_fold_val, preds, average="weighted")

    # Store metrics for the current fold
    fold_metrics.append(
        {
            "fold": fold + 1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

# Calculate average metrics across all folds
avg_accuracy = np.mean([f["accuracy"] for f in fold_metrics])
avg_precision = np.mean([f["precision"] for f in fold_metrics])
avg_recall = np.mean([f["recall"] for f in fold_metrics])
avg_f1 = np.mean([f["f1"] for f in fold_metrics])

# Print out the results
print(f"Avg Accuracy: {avg_accuracy}")
print(f"Avg Precision: {avg_precision}")
print(f"Avg Recall: {avg_recall}")
print(f"Avg F1 Score: {avg_f1}")
