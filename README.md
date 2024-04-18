# Welcome! Pay Attention Cause That's All You Need

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
  - [attention_classifier.py](#attention-classifierpy)
  - [preprocess_text.py](#preprocess-textpy)
- [Usage](#usage)
- [Setup](#setup)
- [Dependencies](#dependencies)
- [Driver Notebook](#driver-notebook)
- [Results Visualization](#results-visualization)
- [License](#license)

## Introduction

This project showcases the implementation of a simple yet effective attention classifier with a single self-attention head. We train and evaluate this model on the IMDb movie review dataset for sentiment classification tasks. Additionally, we provide a state-of-the-art baseline using BERT attention classifier for comparison.

The primary objective of this project is to illustrate the efficacy of attention mechanisms in capturing intricate contextual patterns within sequences. Despite the simplicity of employing just one attention head, our model demonstrates the capability to discern and extract nuanced features, thereby showcasing the power of attention mechanisms in natural language processing tasks.

## Files

### attention_classifier.py

This file contains implementations of neural network modules for attention and classification. It includes the following classes:

- `Attention`: Implements an attention mechanism module.
- `AddNorm`: Implements a residual connection with layer normalization.
- `Attention_Classification_Model`: Defines the main text classification model using attention.

### preprocess_text.py

This file includes functions for text preprocessing, tokenization, and vocabulary generation. It provides the following functions:

- `tokenize`: Tokenizes text using spaCy tokenizer.
- `generate_vocabulary`: Generates vocabulary from training data.
- `process_data`: Processes text data by tokenizing and converting to indices.

## Usage
To Run the Vannila Classifer:
Install the requirements.txt and execute the `Vannila_Attention_Classifier.ipynb` Jupyter Notebook file. This notebook contains the code for training, evaluating, and analyzing the Vanilla Attention Classifier on the IMDb movie review dataset.

```python
from attention_classifier import Attention_Classification_Model

# Create an instance of the Attention_Classification_Model
model = Attention_Classification_Model(vocab_size, embed_size, hidden_size)

# Now you can use the model for other tasks, such as inference on new data
```
To Run the BERT Classifer:
Install the requirements.txt and execute the `BERT_Classifier.ipynb` Jupyter Notebook file. This notebook contains the code for training, evaluating, and analyzing the BERT Classifier on the IMDb movie review dataset.

## Setup

Explain how to set up your project. Include installation instructions and any necessary configuration steps.

## Dependencies

This project requires the following dependencies:

- Python (>=3.6)
- See [requirements.txt](requirements.txt) for Python package dependencies.

## Driver Notebooks

The driver notebooks `Vannila_Attention_Classifier.ipynb` and `BERT_Classifier.ipynb` provides an example of how to use the implemented modules and train the text classification model. It demonstrates:

- Loading and preprocessing the IMDb dataset.
- Training the vanilla attention-based classification model and BERT model respectively.
- Evaluating the model's performance on validation and test sets.

## Results Visualization

The notebook includes visualization of training and validation loss, training and validation accuracy, and a Receiver Operating Characteristic (ROC) curve.


