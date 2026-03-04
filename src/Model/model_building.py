import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import logging
import yaml
import pandas as pd
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# ---------------- LOGGING ---------------- #

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------- UTIL FUNCTIONS ---------------- #

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug("Parameters loaded successfully from %s", params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug("Data loaded successfully from %s", file_path)
    return df


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


# ---------------- METRICS ---------------- #

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


# ---------------- TRAINING ---------------- #

def train_distilbert(train_df, test_df, params):

    model_name = params["distilbert"]["model_name"]
    num_labels = params["distilbert"]["num_labels"]
    epochs = params["distilbert"]["num_train_epochs"]
    lr = float(params["distilbert"]["learning_rate"])
    batch_size = params["distilbert"]["batch_size"]
    weight_decay = params["distilbert"]["weight_decay"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(
            example["Comment"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = Dataset.from_pandas(train_df[["Comment", "sentiment_encoded"]])
    test_dataset = Dataset.from_pandas(test_df[["Comment", "sentiment_encoded"]])

    train_dataset = train_dataset.rename_column("sentiment_encoded", "labels")
    test_dataset = test_dataset.rename_column("sentiment_encoded", "labels")

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    logger.info("Starting DistilBERT training...")
    trainer.train()

    logger.info("Training completed successfully")

    return trainer


# ---------------- MAIN ---------------- #

def main():
    try:
        root_dir = get_root_directory()

        params = load_params(os.path.join(root_dir, 'params.yaml'))

        train_data = load_data(
            os.path.join(root_dir, 'data/interim/train_processed.csv')
        )

        test_data = load_data(
            os.path.join(root_dir, 'data/interim/test_processed.csv')
        )

        trainer = train_distilbert(train_data, test_data, params)

        # Save trained model
        model_output_path = os.path.join(root_dir, "distilbert_model")
        trainer.save_model(model_output_path)

        # --- SAVE TOKENIZER TOO --- #
        model_name = params["distilbert"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_output_path)

        logger.info("DistilBERT model AND tokenizer saved successfully at %s", model_output_path)

    except Exception as e:
        logger.critical("Model building pipeline failed: %s", e)
        print(f"Fatal Error: {e}")


if __name__ == '__main__':
    main()