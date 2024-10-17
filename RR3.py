import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

# Disable WandB
os.environ["WANDB_DISABLED"] = "true"

def load_and_preprocess_data(file_path, columns_to_keep):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    return df[columns_to_keep]

def balance_dataset(df):
    """Balance the dataset using upsampling."""
    positive_reviews = df[df['reviews.rating'] >= 4]
    neutral_reviews = df[df['reviews.rating'] == 3]
    negative_reviews = df[df['reviews.rating'] <= 2]

    max_class_size = max(len(positive_reviews), len(neutral_reviews), len(negative_reviews))
    positive_upsampled = resample(positive_reviews, replace=True, n_samples=max_class_size, random_state=42)
    neutral_upsampled = resample(neutral_reviews, replace=True, n_samples=max_class_size, random_state=42)
    negative_upsampled = resample(negative_reviews, replace=True, n_samples=max_class_size, random_state=42)

    df_balanced = pd.concat([positive_upsampled, neutral_upsampled, negative_upsampled]).sample(frac=1, random_state=42)
    df_balanced['labels'] = df_balanced['reviews.rating'].map(lambda rating: 0 if rating <= 2 else (1 if rating == 3 else 2))
    return df_balanced

def prepare_datasets(df):
    """Prepare train and test datasets."""
    X_train, X_test, y_train, y_test = train_test_split(
        df['reviews.text'], df['labels'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def tokenize_data(tokenizer, texts, max_length=512):
    """Tokenize the input texts."""
    return tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

def create_hf_dataset(encodings, labels):
    """Create a HuggingFace Dataset."""
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist(),
        'labels': labels.tolist()
    })

def setup_model_and_trainer(num_labels, lora_config, training_args, train_dataset):
    """Set up the DistilBERT model with LoRA and initialize the Trainer."""
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    model = get_peft_model(model, lora_config)
    return Trainer(model=model, args=training_args, train_dataset=train_dataset)

def run_cross_validation(trainer, X_train, y_train, tokenizer, n_splits=3):
    """Run cross-validation and return fold metrics."""
    kf = KFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Running Fold {fold + 1}")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_train_encodings = tokenize_data(tokenizer, X_fold_train)
        fold_val_encodings = tokenize_data(tokenizer, X_fold_val)

        fold_train_dataset = create_hf_dataset(fold_train_encodings, y_fold_train)
        fold_val_dataset = create_hf_dataset(fold_val_encodings, y_fold_val)

        trainer.train_dataset = fold_train_dataset
        trainer.eval_dataset = fold_val_dataset

        trainer.train()
        predictions = trainer.predict(fold_val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy_score(y_fold_val, preds),
            'precision': precision_score(y_fold_val, preds, average='weighted'),
            'recall': recall_score(y_fold_val, preds, average='weighted'),
            'f1': f1_score(y_fold_val, preds, average='weighted')
        })

    return fold_metrics

def print_average_metrics(fold_metrics):
    """Print average metrics across all folds."""
    avg_metrics = {metric: np.mean([f[metric] for f in fold_metrics]) 
                   for metric in ['accuracy', 'precision', 'recall', 'f1']}
    
    for metric, value in avg_metrics.items():
        print(f"Avg {metric.capitalize()}: {value:.4f}")

def main():
    # Load and preprocess data
    file_path = '/notebooks/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
    columns_to_keep = ['name', 'brand', 'primaryCategories', 'reviews.text', 'reviews.rating']
    df = load_and_preprocess_data(file_path, columns_to_keep)
    df_balanced = balance_dataset(df)

    # Prepare datasets
    X_train, X_test, y_train, y_test = prepare_datasets(df_balanced)

    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenize_data(tokenizer, X_train)
    train_dataset = create_hf_dataset(train_encodings, y_train)

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=1,
        no_cuda=False
    )

    # Setup model and trainer
    trainer = setup_model_and_trainer(3, lora_config, training_args, train_dataset)

    # Run cross-validation
    fold_metrics = run_cross_validation(trainer, X_train, y_train, tokenizer)

    # Print average metrics
    print_average_metrics(fold_metrics)

if __name__ == "__main__":
    main()