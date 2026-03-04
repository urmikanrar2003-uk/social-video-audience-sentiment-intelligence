import numpy as np
import pandas as pd
import os
import logging
import re

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ---------------- LABEL MAPPING ---------------- #

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map sentiment_encoded:
    -1 -> 2
     0 -> 0
     1 -> 1
    """
    try:
        label_mapping = {-1: 2, 0: 0, 1: 1}

        if 'sentiment_encoded' not in df.columns:
            raise KeyError("Column 'sentiment_encoded' not found in dataframe.")

        df['sentiment_encoded'] = df['sentiment_encoded'].map(label_mapping)

        logger.debug("Label mapping applied successfully (-1→2, 0→0, 1→1)")

        return df
    except Exception as e:
        logger.error(f"Error during label mapping: {e}")
        raise
# ---------------- BASIC CLEANING ---------------- #

def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for transformer models:
    - Drop NA
    - Remove empty comments
    - Remove URLs
    - Normalize whitespace
    - Remove newline characters
    """
    try:
        df.dropna(inplace=True)

        if 'Comment' not in df.columns:
            raise KeyError("Column 'Comment' not found in dataframe.")

        # Remove empty comments
        df = df[df['Comment'].str.strip() != '']

        # Apply light cleaning
        def clean_text(comment):
            comment = str(comment)
            comment = comment.replace("\n", " ")
            comment = re.sub(r'http\S+|www\S+|https\S+', '', comment)
            comment = re.sub(r'\s+', ' ', comment).strip()
            return comment

        df['Comment'] = df['Comment'].apply(clean_text)

        logger.debug("Minimal transformer-safe cleaning completed")

        return df

    except Exception as e:
        logger.error(f"Error during minimal cleaning: {e}")
        raise

# ---------------- SAVE DATA ---------------- #

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


# ---------------- MAIN ---------------- #

def main():
    try:
        logger.debug("Starting data preprocessing for DistilBERT...")

        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        logger.debug("Raw data loaded successfully")

        # Minimal cleaning
        train_data = minimal_cleaning(train_data)
        test_data = minimal_cleaning(test_data)

        # Apply label mapping
        train_data = map_labels(train_data)
        test_data = map_labels(test_data)

        # Save processed data
        save_data(train_data, test_data, data_path='./data')

        logger.info("Data preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()