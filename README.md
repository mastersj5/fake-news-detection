# Fake News Detection System

## Table of Contents
- [Overview](#overview)
- [Team Members](#team-members)
- [Repository Contents](#repository-contents)
- [Project Pipeline](#project-pipeline)
    - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Feature Engineering & Model Selection](#feature-engineering--model-selection)
    - [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Challenges & Future Work](#challenges--future-work)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to develop a machine learning-based system capable of classifying news articles as either "true" or "fake." We utilize a TensorFlow neural network model in conjunction with natural language processing (NLP) techniques to analyze textual content and identify patterns indicative of misinformation.

**Team Members:**
* Joshua Masters (Primary Developer)
* Samuel Do (Data Preprocessing & Visualization)
* Jake Jeske (Model Optimization & Evaluation)

**Course:** CPS 349 - Data Science

**Instructor:** Dr. Bayley King

## Repository Contents

* `True.csv` & `Fake.csv`: Original datasets containing true and fake news articles.
* `CPS349FinalProject.ipynb`: Jupyter Notebook containing the core project code.
* `README.md`: This documentation file.

## Project Pipeline

1. **Data Loading and Preprocessing:**
   - Import and clean news articles from Kaggle datasets.
   - Focus on articles from 2016-2017 for balanced representation.
   - Standardize date formats, remove duplicates, and introduce a "real" label.
   - Text cleaning: Remove punctuation, special characters, hyperlinks, and convert to lowercase.

2. **Exploratory Data Analysis (EDA):**
   - Generate a word cloud to visualize common words in true and fake articles.
   - Employ spaCy for tokenization, lemmatization, and stop word removal.

3. **Feature Engineering & Model Selection:**
   - Utilize TensorFlow Hub's pre-trained word embeddings to represent textual data.
   - Construct a sequential neural network model with a dense hidden layer and a binary output layer.

4. **Model Training and Evaluation:**
   - Train the model using a binary cross-entropy loss and the Adam optimizer.
   - Evaluate model performance on a held-out test set using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
   - Visualize training and validation loss/accuracy to monitor convergence and potential overfitting.

## Results

The initial model achieved an accuracy of [0.7000] on the test set. Further analysis using the confusion matrix revealed insights into the model's strengths and weaknesses in classifying specific types of articles.

## Challenges & Future Work

- **Data Imbalance:** The datasets may have a class imbalance, which can affect model performance. Techniques like SMOTE or cost-sensitive learning could be explored to address this.
- **Hyperparameter Tuning:** Optimize the model architecture (number of layers, neurons per layer), learning rate, batch size, and other hyperparameters to improve performance.
- **Alternative Models:** Experiment with different models like Support Vector Machines (SVM), Random Forests, or BERT-based architectures.
- **Real-World Data:** Test the model on a wider range of news sources and evaluate its robustness to different styles and topics.

## Getting Started

1. Clone this repository.
2. Install the required libraries: `pip install -r requirements.txt` 
3. Run the Jupyter Notebook: `jupyter notebook CPS349FinalProject.ipynb`

## Acknowledgements

* Dataset Source: [Kaggle Dataset Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/)
* TensorFlow Hub: [TensorFlow Hub Link](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)