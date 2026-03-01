import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier

import random

np.random.seed(42)
random.seed(42)


# ---------------- LOGGING ---------------- #

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------- UTIL FUNCTIONS ---------------- #

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded successfully from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("Params file not found at %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params: %s", e)
        raise



def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple):
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = train_data['Comment'].values
        y_train = train_data['sentiment_encoded'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug("TF-IDF transformation complete. Shape: %s", X_train_tfidf.shape)

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF vectorizer saved successfully")

        return X_train_tfidf, y_train

    except KeyError as e:
        logger.error("Column missing in dataset: %s", e)
        raise
    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise

# ---------------- STACKING TRAINING ---------------- #

def train_stacking(X_train, y_train, params):
    try:

        gb_params = params["stacking"]["base_models"]["gradient_boosting"]
        cb_params = params["stacking"]["base_models"]["catboost"]
        lr_params = params["stacking"]["base_models"]["logistic"]
        nb_params = params["stacking"]["base_models"]["naive_bayes"]
        final_params = params["stacking"]["final_estimator"]

        base_models = [
            ("gb", GradientBoostingClassifier(
                n_estimators=gb_params["n_estimators"],
                learning_rate=gb_params["learning_rate"],
                max_depth=gb_params["max_depth"],
                random_state=42
            )),
            ("cb", CatBoostClassifier(
                depth=cb_params["depth"],
                learning_rate=cb_params["learning_rate"],
                iterations=cb_params["iterations"],
                verbose=0,
                random_state=42
            )),
            ("lr", LogisticRegression(
                C=lr_params["C"],
                max_iter=1000,
                random_state=42
            )),
            ("nb", MultinomialNB(
                alpha=nb_params["alpha"],
            ))
        ]

        final_estimator = LogisticRegression(
            C=final_params["C"],
            max_iter=1000
        )

        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=params["stacking"]["cv"],
            n_jobs=-1,
        )

        stacking_model.fit(X_train, y_train)

        logger.debug("Stacking model training completed")
        return stacking_model
    except KeyError as e:
        logger.error("Missing parameter in YAML file: %s", e)
        raise
    except Exception as e:
        logger.error("Error during stacking model training: %s", e)
        raise

def save_model(model, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved successfully at %s", file_path)
    except Exception as e:
        logger.error("Error while saving model: %s", e)
        raise

# ---------------- MAIN ---------------- #

def main():
    try:
        root_dir = get_root_directory()

        params = load_params(os.path.join(root_dir, 'params.yaml'))

        max_features = params['vectorizer']['max_features']
        ngram_range = tuple(params['vectorizer']['ngram_range'])

        train_data = load_data(
            os.path.join(root_dir, 'data/interim/train_processed.csv')
        )

        X_train_tfidf, y_train = apply_tfidf(
            train_data,
            max_features,
            ngram_range
        )

        stacking_model = train_stacking(X_train_tfidf, y_train, params)

        save_model(
            stacking_model,
            os.path.join(root_dir, 'stacking_model.pkl')
        )

        logger.info("Model building pipeline completed successfully")

    except Exception as e:
        logger.critical(
            "Model building pipeline failed: %s",
            e
        )
        print(f"Fatal Error: {e}")

    

if __name__ == '__main__':
    main()