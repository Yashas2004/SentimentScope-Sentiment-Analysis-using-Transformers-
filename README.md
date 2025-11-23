📘 SentimentScope: Sentiment Analysis using Transformers

SentimentScope is an end-to-end sentiment analysis project built using Transformers and the IMDB Movie Reviews dataset.
This project demonstrates data loading, preprocessing, model training, evaluation, and insights using state-of-the-art NLP techniques.

🚀 Project Overview

This project performs binary sentiment classification (positive / negative) using text reviews from the IMDB dataset.
It showcases how to:

Load and preprocess raw text data

Use modern Transformer models for NLP tasks

Train a classifier on the IMDB dataset

Evaluate model performance

Prepare datasets for custom workflows

📂 Dataset: IMDB Movie Reviews

The project uses the ACL IMDB dataset, containing:

25,000 labeled training reviews

25,000 labeled testing reviews

Separate folders for pos and neg

✔ Dataset Structure:
aclImdb/
│── train/
│   ├── pos/
│   └── neg/
│── test/
    ├── pos/
    └── neg/


The notebook automatically extracts the dataset using:

!tar -xzf aclImdb_v1.tar.gz


Then loads positive and negative samples from respective folders.

🧹 Data Loading & Preprocessing

A helper function load_dataset() reads all .txt files from a given directory and stores them into Python lists.

def load_dataset(folder):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename)) as f:
                texts.append(f.read())
    return texts


The dataset is then split and prepared for model training.

🤖 Modeling with Transformers

This project uses a Transformer-based architecture (e.g., BERT, DistilBERT, etc.) for sentiment classification.

Steps included:

Tokenization

Encoding text into model-friendly format

Training a classification head

Evaluating accuracy on test data

This ensures high-quality predictions and low pre-processing overhead.

📊 Evaluation

The notebook includes scripts to evaluate performance metrics such as:

Accuracy

Loss curves

Classification quality

Plots and metrics help in understanding how well the model performs.

🧪 Tech Stack
Component	Technology
Language	Python
Dataset	IMDB Movie Reviews
ML Model	Transformer Models (HuggingFace)
Libraries	PyTorch / TensorFlow, Transformers, Pandas, NumPy
Notebook	Jupyter / Google Colab
📁 Project Structure
project/
│── SentimentScope.ipynb        # Main notebook
│── aclImdb_v1.tar.gz           # Dataset archive
│── README.md                   # Project documentation

▶ How to Run

Clone the repo:

git clone [https://github.com/Yashas2004/SentimentScope-Sentiment-Analysis-using-Transformers.git]


Install dependencies:

pip install transformers torch pandas numpy


Extract dataset:

tar -xzf aclImdb_v1.tar.gz


Open the notebook:

jupyter notebook SentimentScope.ipynb


Run all cells!


📝 License

This project is open-source and available under the MIT License.
