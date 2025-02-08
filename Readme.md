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

## Results & Discussion

The training script generates plots for each model that include:

- **Accuracy Curves:**  
  Both training and validation accuracy are plotted over epochs.

  - **Observation:** The Advanced DNN model often shows improved convergence compared to the Simple DNN, indicating that additional layers and normalization help capture complex patterns in the data.

- **Loss Curves:**  
  Training and validation loss curves illustrate how quickly the models reduce error over time.

  - **Observation:** A steeper drop in loss typically suggests a model that is learning effectively. The BERT model’s loss often decreases faster and to a lower value, which is common for transfer learning approaches with contextual embeddings.

- **Comparison Summary:**  
  Along with plots, evaluation metrics (accuracy, precision, recall, F1-score) are printed for each model.
  - **Observation:** The BERT model frequently outperforms both DNN models, highlighting the advantage of leveraging pre-trained language models for sentiment analysis.

These visualizations and metrics help in understanding each model’s strengths and weaknesses, guiding future improvements and model selection based on the desired trade-offs between performance and computational cost.

## Conclusion

This project provides a framework for benchmarking different sentiment analysis models. It shows how traditional machine learning methods (using TF-IDF and DNNs) compare to modern transfer learning techniques (using BERT). The repository serves as a starting point for further experimentation, including hyperparameter tuning and exploring additional architectures.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- IMDb dataset provided by Kaggle.
- Transformer implementation powered by [Hugging Face Transformers](https://huggingface.co/transformers/).
