```markdown
# Image Captioning Project

## Overview

This project aims to generate textual descriptions (captions) for images using machine learning and natural language processing techniques. It utilizes the Flickr8k dataset and implements a CNN-LSTM architecture for image captioning.

## How to Run

1. **Clone the Repository:** 
   ```bash
   git clone https://github.com/rishabhshah13/image-captioning.git
   cd image-captioning
   ```

2. **Install Dependencies:** 
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:** 
   - Download the Flickr8k dataset from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).
   - Extract the dataset into the `data` directory.

4. **Training:**
   - Run [TRAIN-image-captioning-with-attention.ipynb](TRAIN-image-captioning-with-attention.ipynb) to train the captioning model with attention.
   - Run [TRAIN-image-captioning-with-out-attention.ipynb](TRAIN-image-captioning-with-out-attention.ipynb) to train the captioning model without attention.

5. **Inference:**
   - Run [Inference-image-captioning-with-attention.ipynb](Inference-image-captioning-with-attention.ipynb) to perform inference using the saved model with attention.
   - Run [Inference-image-captioning-with-out-attention.ipynb](Inference-image-captioning-with-out-attention.ipynb) to perform inference using the saved model without attention.

## Dataset

The dataset used for this project is the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). It consists of 8092 images along with descriptive captions for each image.

## Model Architecture

### CNN-LSTM Model

- **CNN Architecture:** ResNet-152, pre-trained on ImageNet
- **LSTM Hidden Size:** 512
- **Word Embedding Size:** 300 (using pre-trained GloVe embeddings)
- **Batch Size:** 64
- **Optimizer:** Adam
- **Learning Rate:** 4e-4 (with decay)
- **Dropout:** 0.5
- **Epochs:** 30-50
- **Early Stopping:** Based on validation loss, patience of 10 epochs

### Encoder-Decoder with Attention Model

- **CNN Encoder:** Inception-V3, pre-trained on ImageNet
- **RNN Decoder:** 2-layer LSTM with 512 hidden units
- **Word Embedding Size:** 512 (using pre-trained GloVe embeddings)
- **Attention Mechanism:** Bahdanau Attention
- **Batch Size:** 32
- **Optimizer:** Adam
- **Learning Rate:** 1e-4 (with decay)
- **Dropout:** 0.3 for LSTM, 0.5 for attention
- **Epochs:** 40-60
- **Early Stopping:** Based on validation loss, patience of 15 epochs
- **Teacher Forcing Ratio:** 0.5 (ratio of using ground truth vs. predicted words)
- **Beam Search:** During inference, with a beam size of 5
```
Feel free to add more sections or details as needed!
