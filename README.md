# Image Caption Generator with Custom CNN-LSTM

A complete deep learning pipeline for generating natural language captions from images, built from scratch using a custom Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network. This project demonstrates end-to-end image captioning, from data preprocessing to model evaluation, using only foundational deep learning components—no pre-trained backbones.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training & Hyperparameter Tuning](#training--hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Results](#sample-results)
- [How to Run](#how-to-run)
- [Project Team](#project-team)
- [References](#references)

---

## Project Overview

This project implements an **image captioning system** that generates fluent, context-aware captions for images. The approach leverages a two-branch neural architecture:
- **Custom CNN**: Extracts rich visual features from input images.
- **LSTM Decoder**: Generates captions, conditioned on the extracted features.

The pipeline is built and trained from scratch, focusing on transparency, reproducibility, and educational clarity for deep learning practitioners.

---

## Dataset

- **Size**: 8,091 images, each annotated with 5 human-written captions.
- **Splits**: 
  - Training: 70%
  - Validation: 15%
  - Test: 15%
- **Captions**: Cleaned, lowercased, and tokenized; padded/truncated to a maximum length (95th percentile ≤ 40 tokens).
- **Vocabulary**: ~8,500 unique words.

---

## Data Preprocessing

1. **Caption Cleaning**: Lowercase, remove punctuation, normalize whitespace, and wrap with `startseq` and `endseq` tokens.
2. **Integrity Checks**: Ensure every caption has a corresponding image and vice versa.
3. **Tokenizer**: Fit on all captions to build a word-to-index mapping; pad/truncate sequences to uniform length.
4. **Feature Extraction**: 
   - Images resized to 299×299 pixels.
   - Custom CNN extracts feature vectors (three Conv2D + MaxPooling2D blocks, followed by flattening).
   - Features cached to disk for efficiency.
5. **Training Sequences**: For each caption, generate multiple (image feature, partial caption) → next word pairs for supervised training.

---

## Model Architecture

**Encoder (Image Branch):**
- Input: Flattened CNN feature vector.
- Dropout (0.3) → Dense(256, ReLU).

**Decoder (Text Branch):**
- Input: Caption sequence (as integer tokens).
- Embedding (vocab_size, embedding_dim, mask_zero=True) → Dropout (0.3) → LSTM(256).

**Fusion & Output:**
- Decoder output projected via Dense(256), then added element-wise to encoder output.
- Combined vector → Dense(256, ReLU) → Dense(vocab_size, softmax).

**Loss**: Sparse categorical crossentropy  
**Optimizer**: Adam

---

## Training & Hyperparameter Tuning

- **Grid Search**: Embedding dimension, LSTM units, dropout rate, and learning rate.
- **Cross-Validation**: 5-fold, ensuring no image appears in both train and validation splits for a fold.
- **Early Stopping**: Halts training if validation loss does not improve for 3 epochs.
- **Model Checkpointing**: Saves the best model based on validation loss.

**Best Hyperparameters Identified:**
- Embedding Dimension: 256
- LSTM Units: 512
- Dropout Rate: 0.3
- Learning Rate: 1e-3

---

## Evaluation Metrics

- **Test Loss**: 3.52
- **Test Accuracy**: 36%
- **BLEU-1**: 0.412
- **BLEU-4**: 0.057

BLEU-1 measures unigram overlap (word choice), while BLEU-4 measures up to 4-gram overlap (fluency and structure) between generated and reference captions.

---

## Sample Results

Below are examples of generated captions for random test images (visualizations available in the notebook):

> **Image 1:** "a man riding a wave on top of a surfboard"
>
> **Image 2:** "a group of people standing around a kitchen preparing food"
>
> **Image 3:** "a dog is running through a grassy field"

*Note: Actual generated captions may vary depending on random seed and final model weights.*

---

## How to Run

1. **Clone the Repository**

git clone https://github.com/yourusername/cnn-lstm-image-caption-generator.git
cd cnn-lstm-image-caption-generator


2. **Prepare the Dataset**

- Place images and `captions.txt` in the appropriate directories as referenced in the notebook.
- Ensure directory structure matches the code paths.

3. **Install Dependencies**

pip install tensorflow keras nltk scikit-learn matplotlib tqdm


- Download NLTK punkt tokenizer (first run will auto-download).

4. **Run the Notebook**

- Open `Mini-Project-2021E045108187.ipynb` in Jupyter or VS Code.
- Execute cells sequentially to preprocess data, train the model, and evaluate results.

5. **Model Outputs**

- Model checkpoints, tokenizer, and other artifacts are saved in the `output/` directory.

---

## Project Team

- **H.M.A.Y. Herath** (2021/E/045)
- **P.G.P.M. Chandrasiri** (2021/E/108)
- **H.M.S.A. Bandara** (2021/E/187)

Course: EC9170 – Deep Learning for Electrical & Computer Engineers  
Date: 2025-04-24

---

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Keras Documentation](https://keras.io/)
- [NLTK BLEU Score](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)

---

## Acknowledgements

This project was developed as part of the EC9170 course mini project, focusing on building deep learning solutions from scratch for real-world tasks.
