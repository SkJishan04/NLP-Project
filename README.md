# NLP-Project

This repository contains a project that performs sentiment analysis on Yelp reviews using machine learning techniques. The goal is to classify reviews with 1-star and 5-star ratings using text data. 

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Results](#results)
- [Usage](#usage)

## Project Overview

This project demonstrates the following steps:

1. Exploratory Data Analysis (EDA) using Pandas, Matplotlib, and Seaborn.
2. Preprocessing the Yelp review text data.
3. Building and training machine learning models using Naive Bayes.
4. Evaluating the model's performance using confusion matrix and classification report.

## Dataset

The dataset used in this project is `yelp.csv`, which contains Yelp reviews with the following columns:

- **text**: The review text.
- **stars**: The star rating of the review (1 to 5).

A new feature, `text_length`, is calculated as the length of each review's text.

## Dependencies

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Exploratory Data Analysis

The EDA includes:

1. Visualization of text length distribution across different star ratings.
2. Boxplots to analyze the relationship between review length and star ratings.
3. Count plots to visualize the frequency of different star ratings.
4. Heatmap to explore correlations between numerical features.

## Model Building

Two approaches are used for model building:

1. **Naive Bayes with Bag of Words (CountVectorizer):**
    - Preprocess the text data using CountVectorizer.
    - Train the Naive Bayes model on the vectorized data.

2. **Naive Bayes with TF-IDF Pipeline:**
    - Use a pipeline to combine CountVectorizer, TF-IDF Transformer, and Naive Bayes.
    - Train and evaluate the model on a train-test split.

## Results

The performance of the models is evaluated using:

- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Usage

To use this project:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/nlp-project.git
   cd nlp-project
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `yelp.csv` file is in the same directory.

4. Run the analysis script:

   ```bash
   python analysis.py
   ```

The `analysis.py` script contains the entire workflow from data loading to model evaluation.

---

Feel free to contribute to this project by creating issues or submitting pull requests!

