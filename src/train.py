#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from transformers import BertTokenizer, TFBertForSequenceClassification

# Import functions from our modules.
from data_loader import extract_dataset, load_dataset, split_data
from preprocess import prepare_dataframe
from models import build_simple_dnn, build_advanced_dnn, build_transformer_model, prepare_transformer_data


def setup_gpu():
    """Ensures TensorFlow is using GPU and configures memory growth."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ Using GPU: {physical_devices[0].name}")
    else:
        print("❌ No GPU detected. Running on CPU.")


def prepare_tfidf_features(df, max_features=5000):
    """Converts preprocessed text into TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()
    y = df['label'].values
    return X, y, vectorizer


def plot_history(history, title, filename):
    """Plots training and validation accuracy over epochs and saves it."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"results/{filename}")
    plt.close()


def main():
    # Setup GPU
    setup_gpu()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Define file paths.
    zip_path = "data/imdb-dataset-of-50k-movie-reviews.zip"
    csv_path = "data/IMDB Dataset.csv"

    # Extract dataset if CSV doesn't exist.
    if not os.path.exists(csv_path):
        extract_dataset(zip_path, extract_to='data')

    # Load and preprocess dataset.
    df = load_dataset(csv_path)
    df = prepare_dataframe(df)


    # ---------------------
    # DNN Models using TF-IDF Features
    # ---------------------
    X_tfidf, y, _ = prepare_tfidf_features(df)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test = split_data(X_tfidf, y)
    input_dim = X_train_tfidf.shape[1]

    results = {}

    # Train Model 1
    simple_dnn_model = build_simple_dnn(input_dim)
    history1 = simple_dnn_model.fit(X_train_tfidf, y_train,
                      validation_data=(X_val_tfidf, y_val),
                      epochs=20, batch_size=32)
    plot_history(history1, "Simple DNN Performance", "simple_dnn_model_performance.png")
    test_loss1, test_acc1 = simple_dnn_model.evaluate(X_test_tfidf, y_test)
    results["Simple DNN"] = {"Test Accuracy": test_acc1, "Test Loss": test_loss1}

    # Train Model 2
    advanced_dnn_model = build_advanced_dnn(input_dim)
    history2 = advanced_dnn_model.fit(X_train_tfidf, y_train,
                      validation_data=(X_val_tfidf, y_val),
                      epochs=20, batch_size=32)
    plot_history(history2, "Advanced DNN Performance", "advanced_dnn_model_performance.png")
    test_loss2, test_acc2 = advanced_dnn_model.evaluate(X_test_tfidf, y_test)
    results["Advanced DNN"] = {"Test Accuracy": test_acc2, "Test Loss": test_loss2}

    # ---------------------
    # Transfer Learning Model using BERT
    # ---------------------
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Use MirroredStrategy for single-GPU training
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Prepare input data
        train_encodings = prepare_transformer_data(train_df['review'].tolist(), tokenizer)
        val_encodings = prepare_transformer_data(val_df['review'].tolist(), tokenizer)
        test_encodings = prepare_transformer_data(test_df['review'].tolist(), tokenizer)

        train_labels = np.array(train_df['label'].tolist())
        val_labels = np.array(val_df['label'].tolist())
        test_labels = np.array(test_df['label'].tolist())

        # Create TensorFlow datasets
        batch_size = 16
        train_dataset = tf.data.Dataset.from_tensor_slices(((dict(train_encodings), train_labels))).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(((dict(val_encodings), val_labels))).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((dict(test_encodings), test_labels))).batch(batch_size)

        # Compile model
        optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metric = SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Train the model
        history_bert = bert_model.fit(train_dataset, validation_data=val_dataset, epochs=5)
        plot_history(history_bert, "BERT Model Performance", "bert_performance.png")

        # Evaluate the model
        test_loss_bert, test_acc_bert = bert_model.evaluate(test_dataset)
        results = {"BERT": {"Test Accuracy": test_acc_bert, "Test Loss": test_loss_bert}}

    # Save results to JSON
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
