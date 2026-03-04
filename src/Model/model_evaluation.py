import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import logging
import yaml
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix


# ---------------- LOGGING ---------------- #

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
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
    logger.debug("Parameters loaded successfully")
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug("Test data loaded successfully")
    return df


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def save_model_info(run_id: str, model_path: str, file_path: str):
    model_info = {
        "run_id": run_id,
        "model_path": model_path
    }
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)


# ---------------- MAIN ---------------- #

def main():

    mlflow.set_tracking_uri("http://ec2-3-87-202-243.compute-1.amazonaws.com:5000")
    mlflow.set_experiment("dvc-distilbert-runs")

    with mlflow.start_run() as run:
        try:
            root_dir = get_root_directory()

            # Load params
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for key, value in params["distilbert"].items():
                mlflow.log_param(key, value)

            # Load test data
            test_data = load_data(
                os.path.join(root_dir, 'data/interim/test_processed.csv')
            )

            texts = test_data["Comment"].tolist()
            labels = test_data["sentiment_encoded"].values

            # Load model + tokenizer
            model_path = os.path.join(root_dir, "distilbert_model")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Tokenize
            encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            encodings = {k: v.to(device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = model(**encodings)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Classification report
            report = classification_report(labels, predictions, output_dict=True)
            cm = confusion_matrix(labels, predictions)

            # Log per-class metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"{label}_precision", metrics["precision"])
                    mlflow.log_metric(f"{label}_recall", metrics["recall"])
                    mlflow.log_metric(f"{label}_f1_score", metrics["f1-score"])

            # Log confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=["neutral(0)", "positive(1)", "negative(2)"],
                yticklabels=["neutral(0)", "positive(1)", "negative(2)"]
            )
            plt.title("Confusion Matrix - DistilBERT")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # Log model
            mlflow.pytorch.log_model(model, "distilbert_model")

            # Save experiment info
            artifact_uri = mlflow.get_artifact_uri()
            save_model_info(
                run.info.run_id,
                f"{artifact_uri}/distilbert_model",
                "experiment_info.json"
            )

            mlflow.set_tag("model_type", "DistilBERT")
            mlflow.set_tag("task", "Sentiment Analysis")

            logger.info("Model evaluation completed successfully")

        except Exception as e:
            logger.error("Evaluation pipeline failed: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()