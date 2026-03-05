import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

import os
import re
import io
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─── APP SETUP ───────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# ─── LOGGING ─────────────────────────────────────────────────────────────────

logger = logging.getLogger('flask_api')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('flask_api_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────

# The model is saved as a HuggingFace directory, not a .pkl file.
# It is produced by the model_building DVC stage.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'distilbert_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading DistilBERT tokenizer and model from: %s", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
logger.info("Model loaded successfully on %s", device)

# Sentiment label mapping (matches training: 0=neutral, 1=positive, 2=negative)
LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def minimal_cleaning(texts: list) -> list:
    """
    Minimal transformer-safe cleaning:
    - Remove URLs
    - Remove newline characters
    - Normalize whitespace
    """
    cleaned = []
    for comment in texts:
        comment = str(comment)
        comment = comment.replace("\n", " ")
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()
        cleaned.append(comment)
    logger.debug("Cleaned %d comments", len(cleaned))
    return cleaned


def run_inference(texts: list) -> list:
    """Run DistilBERT inference on a list of text strings. Returns predicted label strings."""
    cleaned_texts = minimal_cleaning(texts)

    encodings = tokenizer(
        cleaned_texts,
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

    return [LABEL_MAP[int(p)] for p in predictions]


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return "Welcome to the Social Video Sentiment Intelligence API"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON: {"comments": ["comment1", "comment2", ...]}
    Returns:      [{"comment": "...", "sentiment": "positive/negative/neutral"}, ...]
    """
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        predictions = run_inference(comments)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """
    Expects JSON: {"comments": [{"text": "...", "timestamp": "2024-01-01"}, ...]}
    Returns:      [{"comment": "...", "sentiment": "...", "timestamp": "..."}, ...]
    """
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments   = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        predictions = run_inference(comments)
    except Exception as e:
        logger.error("Prediction with timestamps failed: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """
    Expects JSON: {"sentiment_counts": {"positive": 10, "neutral": 5, "negative": 3}}
    Returns: PNG pie chart image
    """
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('positive', sentiment_counts.get('1', 0))),
            int(sentiment_counts.get('neutral',  sentiment_counts.get('0', 0))),
            int(sentiment_counts.get('negative', sentiment_counts.get('2', 0)))
        ]

        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Error in /generate_chart: %s", e)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """
    Expects JSON: {"comments": ["comment1", "comment2", ...]}
    Returns: PNG word cloud image
    """
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        cleaned = minimal_cleaning(comments)
        text = ' '.join(cleaned)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Error in /generate_wordcloud: %s", e)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """
    Expects JSON: {"sentiment_data": [{"sentiment": "positive", "timestamp": "2024-01-15"}, ...]}
    Returns: PNG trend graph image
    """
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Map string labels to numeric for resampling (matches training: 0=neutral, 1=positive, 2=negative)
        label_to_num = {"positive": 1, "neutral": 0, "negative": 2}
        df['sentiment'] = df['sentiment'].map(label_to_num).fillna(df['sentiment']).astype(int)

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [0, 1, 2]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0
        monthly_percentages = monthly_percentages[[2, 0, 1]]

        sentiment_labels = {2: 'Negative', 0: 'Neutral', 1: 'Positive'}
        colors = {2: 'red', 0: 'gray', 1: 'green'}

        plt.figure(figsize=(12, 6))
        for val in [2, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker='o', linestyle='-',
                label=sentiment_labels[val],
                color=colors[val]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Error in /generate_trend_graph: %s", e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
