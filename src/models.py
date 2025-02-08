import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Import Hugging Face Transformers for the BERT model.
from transformers import BertTokenizer, TFBertForSequenceClassification

def build_simple_dnn(input_dim):
    """
    Builds a Simple DNN Model for sentiment analysis.
    
    This model consists of:
      - An input layer with 128 neurons (ReLU activation)
      - A Dropout layer (rate=0.5)
      - A hidden layer with 64 neurons (ReLU activation)
      - A Dropout layer (rate=0.5)
      - An output layer with 1 neuron (sigmoid activation for binary classification)
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Simple DNN Model Summary:")
    model.summary()
    return model

def build_advanced_dnn(input_dim):
    """
    Builds an Advanced DNN Model with Batch Normalization for sentiment analysis.
    
    This model consists of:
      - An input layer with 256 neurons (ReLU activation)
      - A Batch Normalization layer
      - A Dropout layer (rate=0.5)
      - A hidden layer with 128 neurons (ReLU activation)
      - A Batch Normalization layer
      - A Dropout layer (rate=0.5)
      - Another hidden layer with 64 neurons (ReLU activation)
      - An output layer with 1 neuron (sigmoid activation for binary classification)
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Advanced DNN Model Summary:")
    model.summary()
    return model

def build_transformer_model(model_name="bert-base-uncased", num_labels=2):
    """
    Builds a transformer model (BERT) for sequence classification.
    Returns both the tokenizer and the model.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def prepare_transformer_data(texts, tokenizer, max_length=256):
    """
    Tokenizes a list of texts for the transformer model.
    """
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf"
    )
