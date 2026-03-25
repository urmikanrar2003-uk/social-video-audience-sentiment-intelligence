# 📊 Social Video Audience Sentiment Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=flat)](https://dvc.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> An automated, end-to-end Machine Learning system that analyzes public opinion and audience sentiment on YouTube streams and videos.

## 🌟 About the Project
Understanding audience reactions manually on large YouTube comment sections is nearly impossible. This project bridges that gap by providing a seamless **Chrome Extension** that interacts with a robust **Machine Learning Backend**. It extracts comments, processes them using state-of-the-art NLP models (DistilBERT/CatBoost), and returns an aggregated sentiment analysis score directly to the user's browser.

### ✨ Key Features
*   **Browser Integration:** A sleek Chrome Extension to analyze YouTube videos on the fly.
*   **Advanced NLP Models:** Leverages fine-tuned transformer models (DistilBERT) and gradient boosting (CatBoost) for high-accuracy sentiment grouping.
*   **Reproducible ML Pipeline:** End-to-end data versioning, training, and evaluation managed entirely by **DVC** and **MLflow**.
*   **Scalable API:** Model inference is served rapidly via a containerized **Flask API**.
*   **CI/CD Ready:** Automated testing and deployment workflows via GitHub Actions.

---

## 🏗 System Architecture & Tech Stack

This project is broken down into three main components: a Machine Learning pipeline, a Backend API, and a Frontend extension.

**1. Machine Learning & Data Pipeline**
*   **Data Scraper & NLP Processing:** Pandas, NLTK, Scikit-Learn.
*   **Modeling:** HuggingFace `transformers` (DistilBERT) and `catboost`.
*   **Pipeline & Tracking:** Managed by **DVC** (Data Version Control) alongside **MLflow** for experiment tracking and metric logging.

**2. Backend (Inference Service)**
*   **Framework:** Built entirely on **Flask**.
*   **Containerization:** Fully containerized using **Docker** for standardized deployment across environments.
*   **CI/CD:** Automated builds and tests via **GitHub Actions**.

**3. Frontend User Interface**
*   **YouTube Integration:** A native **Google Chrome Extension**.
*   **Technologies:** HTML, CSS, JavaScript (via manifest V3).

---

## 📂 Project Structure

A high-level overview of the repository's layout:

```text
social-video-audience-sentiment-intelligence/
├── .github/workflows/         # CI/CD pipelines (GitHub Actions)
├── data/                      # Raw and processed datasets (tracked by DVC)
├── distilbert_model/          # DistilBERT transformer models/weights
├── dvc.yaml & params.yaml     # DVC pipeline stages and hyperparameters
├── flask_api/                 # Flask backend inference API
│   └── main.py                
├── mlruns/                    # MLflow experiment tracking logs
├── src/                       # Core ML source code (preprocessing, train, eval)
├── yt-chrome-plugin-frontend/ # Chrome Extension frontend (HTML, JS, Manifest)
├── dockerfile                 # Docker configuration for API deployment
├── requirements.txt           # Python application dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to set up the project locally. 

### Prerequisites
*   [Python 3.11+](https://www.python.org/downloads/)
*   [Git](https://git-scm.com/)
*   [Docker](https://www.docker.com/) (Optional, for containerized run)
*   Google Chrome Browser

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/social-video-audience-sentiment-intelligence.git
cd social-video-audience-sentiment-intelligence
```

### 2. Environment Setup & Dependencies
It is highly recommended to use a virtual environment.
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Backend API
You can run the Flask inference API either natively or via Docker.

**Option A: Natively (Python)**
```bash
python flask_api/main.py
```
*The API will typically start on `http://127.0.0.1:5000`.*

**Option B: Using Docker**
```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

### 4. Installing the Chrome Extension
To use the UI frontend on YouTube:
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle in the top right corner.
3. Click the **Load unpacked** button in the top left.
4. Select the `yt-chrome-plugin-frontend` folder from this repository.
5. The extension is now installed! You can pin it to your browser toolbar for easy access.

---

## 🧠 Machine Learning Pipeline

This project relies on **Data Version Control (DVC)** to manage entire machine learning pipelines, ensuring that data processing and model training are reproducible. 

The pipeline is defined in `dvc.yaml` and consists of five sequential stages:
1.  **Data Ingestion:** Extracts raw data logic (`src/data/data_ingestion.py`).
2.  **Data Preprocessing:** Cleans and formats the text, splitting it into train/test sets.
3.  **Model Building:** Trains the DistilBERT/CatBoost models saving the weights to the `distilbert_model/` directory.
4.  **Model Evaluation:** Generates performance metrics and saves them to `experiment_info.json`.
5.  **Model Registration:** Registers the best performing model via MLflow.

**To run the entire pipeline end-to-end, simply execute:**
```bash
dvc repro
```

*Want to visualize the experiment metrics? Run the MLflow UI local server:*
```bash
mlflow ui
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! 
If you have suggestions to improve this project, please fork the repository and create a pull request.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.
