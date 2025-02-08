# Sentiment-Model-Benchmark

## Overview

This repository benchmarks multiple sentiment analysis models on the IMDb movie reviews dataset. It compares three different approaches:

- **Simple DNN Model:** A basic dense neural network that uses TF-IDF features.
- **Advanced DNN Model:** A deeper network enhanced with batch normalization to capture more complex patterns.
- **Transformer (BERT) Model:** A state-of-the-art transfer learning model fine-tuned for sentiment classification.

The goal is to evaluate and compare these methods, providing insights into their training dynamics and overall performance.

## Repository Structure

```
Sentiment-Model-Benchmark/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # (Dataset files are stored or extracted here)
└── src/
    ├── __init__.py
    ├── data_loader.py     # Data extraction, loading, and splitting functions
    ├── preprocess.py      # Text cleaning and preprocessing functions
    ├── models.py          # Model definitions (Simple DNN, Advanced DNN, BERT)
    └── train.py           # Main script for training and evaluating models
```

## Dataset

The project uses the IMDb movie reviews dataset (50K reviews labeled as "positive" or "negative").  
If the dataset CSV file is not already present in the `data/` folder, the training script will automatically extract it from the provided zip file.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Sentiment-Model-Benchmark.git
   cd Sentiment-Model-Benchmark
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Prepare the Dataset:**

   Ensure you have the IMDb dataset zip file (e.g., `imdb-dataset-of-50k-movie-reviews.zip`) in the expected location. If necessary, update the zip file path in `src/train.py`.

2. **Run the Training Script:**

   ```bash
   python src/train.py
   ```

   The script will:

   - Load and preprocess the data.
   - Extract TF-IDF features for the DNN models.
   - Build, train, and evaluate the Simple DNN and Advanced DNN models (each trained for 10 epochs).
   - Fine-tune and evaluate the BERT model (trained for 3 epochs).
   - Plot training/validation accuracy and loss curves for all models.

## Model Training Details

- **Simple DNN Model:**  
  Uses TF-IDF features from preprocessed text and trains for **20 epochs**. Serves as a baseline.

- **Advanced DNN Model:**  
  Incorporates additional layers and batch normalization for enhanced performance; also trained for **20 epochs**.

- **Transformer (BERT) Model:**  
  Fine-tuned using raw text for **5 epochs**. Leverages pre-trained language representations to achieve superior performance.

### Model Comparison: Fine-Tuned BERT vs. Deep Neural Networks (DNNs)

In comparing the simple DNN and the advanced DNN, we observe that both models exhibit signs of overfitting but in different ways. Overfitting occurs when a model performs exceptionally well on the training data but fails to generalize effectively to unseen validation data, as shown by the diverging training and validation accuracy and loss curves.

#### Overfitting Trends:

- **Simple DNN** :

  - **Accuracy**: The model shows an almost perfect training accuracy (~99%), while the validation accuracy plateaus around 88%. This large gap indicates that the model is memorizing the training data and not generalizing well to new data.
  - **Loss**: The training loss decreases consistently, but the validation loss increases after the early epochs, further highlighting overfitting.

- **Advanced DNN** :

  - **Accuracy**: Similar to Simple DNN, Advanced DNN achieves near-perfect accuracy on the training data, but the validation accuracy stagnates at ~88%.
  - **Loss**: While training loss decreases, validation loss starts increasing after the initial epochs, reflecting a similar pattern of overfitting as in Simple DNN.

Both models demonstrate the common symptom of **overfitting**, where the model fits well to the training data but struggles to generalize to unseen data. The most likely causes include model complexity, insufficient regularization, or overtraining.

#### BERT Model:

In contrast, the **fine-tuned BERT** model leverages **transfer learning**, which is particularly effective for tasks like sentiment analysis. BERT, trained on a vast corpus of text, provides contextual embeddings that help the model understand the nuances of language, making it better equipped to generalize to unseen data compared to traditional DNNs.

- **Training and Validation Curves**: The BERT model demonstrates faster convergence and a greater reduction in loss. Unlike the DNN models, BERT's loss continues to decrease to a lower value and maintains a more consistent gap between training and validation curves, suggesting less overfitting.

- **Performance Metrics**:
  - **Accuracy**: BERT outperforms both DNN models with higher accuracy in sentiment classification tasks.
  - **Precision/Recall**: The precision and recall values for BERT are more balanced, as the model performs better in both identifying positive sentiment and avoiding false positives.
  - **F1-Score**: BERT achieves a higher F1-score than both DNN models, indicating a better balance between precision and recall.

#### Conclusion:

While both Simple DNN and Advanced DNN exhibit overfitting, **BERT** demonstrates superior performance due to its use of pre-trained contextual embeddings and more effective generalization. The **BERT model’s ability to learn from a broader understanding of language** allows it to avoid the severe overfitting seen in the DNN models, which tend to memorize the training data rather than learn generalized features. As a result, **BERT is the better model for sentiment analysis**, achieving better overall performance and generalization despite the overfitting observed in the DNN models.

### Key Takeaways:

- **BERT** consistently outperforms DNN models by leveraging pre-trained embeddings and transfer learning for better generalization.
- **DNNs (Simple DNN & Advanced DNN)** exhibit overfitting, as seen in their high training accuracy and stagnant or declining validation accuracy and loss.
- Regularization techniques such as dropout, L2 regularization, and early stopping can help mitigate overfitting in DNN models, but BERT’s transfer learning approach provides a more robust and scalable solution.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- IMDb dataset provided by Kaggle.
- Transformer implementation powered by [Hugging Face Transformers](https://huggingface.co/transformers/).
