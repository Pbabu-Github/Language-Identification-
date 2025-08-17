# Language Identification with Naive Bayes

This project implements a **Naive Bayes language identification model** that predicts the language of a given text using **character n-grams**. The model is trained on sentence-language pairs and can evaluate accuracy and display a confusion matrix.

---

## Features
- Supports multiple languages: English, Russian, Italian, Spanish, French, Turkish, German, Mandarin (`["eng", "rus", "ita", "spa", "fra", "tur", "deu", "cmn"]`)
- Character n-gram based feature extraction
- Naive Bayes model with Laplace smoothing
- Evaluate model accuracy
- Confusion matrix visualization
- Supports sampling of training data for faster experimentation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Pbabu-Github/Language-Identification.git
cd Language-Identification
```

Project Structure

util.py – Utility functions for loading data, generating character n-grams, computing probabilities, and printing confusion matrices.

model.py – Implementation of the Naive Bayes language identification model.

scoring.py – Functions to compute accuracy and confusion matrices.

test.py – Script for training and testing the model on a dataset.

Usage:

```bash
python test.py data/train.tsv data/test.tsv --ngram_size 2 --avg_samples_per_language 100
```

Arguments:

train_file_path: Path to the training data (TSV file)

test_file_path: Path to the test data (TSV file)

--avg_samples_per_language: Optional. Number of examples per language to sample (useful for debugging)

--ngram_size: Size of character n-grams (default is 2)

Example Output

After running test.py, you will see:

Accuracy score of the model

Confusion matrix showing predicted vs actual languages

Example Usage in Code:
```bash
from model import NBLangIDModel
from util import load_data, LANGUAGES
from scoring import accuracy_score, confusion_matrix, print_confusion_matrix

# Load sample data
train_sentences, train_labels = load_data("data/train.tsv", avg_samples_per_language=100)
test_sentences, test_labels = load_data("data/test.tsv", avg_samples_per_language=100)

# Train model
model = NBLangIDModel(ngram_size=2)
model.fit(train_sentences, train_labels)

# Predict
predictions = model.predict(test_sentences)

# Evaluate
print("Accuracy:", accuracy_score(test_labels, predictions))
print_confusion_matrix(confusion_matrix(test_labels, predictions, LANGUAGES), LANGUAGES)

```
Methodology

Load Data: TSV file with sentences and their language labels.

Feature Extraction: Convert sentences into character n-grams.

Train Model: Compute prior probabilities and likelihoods using Naive Bayes with Laplace smoothing.

Predict: Calculate log-probabilities for each language and select the most likely.

Evaluate: Compute accuracy and confusion matrix to assess performance.