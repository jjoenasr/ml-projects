# ML Projects

This repo contains various machine learning projects, each focused on a different technique or application. The projects are organized into the following sections:

- **Exploratory Data Analysis (EDA)**
- **Traditional Supervised Learning**
- **Traditional Unsupervised Learning**
- **Feedforward Neural Networks (FNN)**
- **Convolutional Neural Networks (CNN)**
- **Recurrent Neural Networks (RNN)**
- **Reinforcement Learning (RL)**
- **Generative AI (GenAI)**

Each section contains Jupyter notebooks with code, explanations, and results for specific machine learning tasks.

---

## EDA (Exploratory Data Analysis)
   
1. **[Netflix EDA](EDA/Netflix/NetflixAnalysis.ipynb)**  
   - In this notebook, the Netflix dataset is analyzed, viewing trends, and content metadata.

1. **[Fastfood EDA](EDA/FastFood/TopFastFoodAnalysis.ipynb)**  
   - This notebook explores a fast food dataset, performing analysis on the top worldwide distributors.

---

## Traditional Supervised Learning

1. **[House Pricing Regression](Supervised/Regression/HousingPrice.ipynb)**  
   - A regression problem predicting house prices based on features such as location, square footage, and more.

1. **[Twitter Sentiment Analysis](Supervised/Classification/TwitterSentimentAnalysis.ipynb)**  
   - Analyzing the sentiment of tweets using traditional supervised learning techniques to classify tweets as positive, negative, or neutral.

---

## Traditional Unsupervised Learning

1. **[K-Means Image Compression](Unsupervised/KMeans/ImageCompression.ipynb)**  
   - A notebook demonstrating image compression using k-means clustering to reduce the number of colors in an image.

2. **[Dimensionality Reduction of Embeddings](Unsupervised/DimReduction/EmbeddingVisualization.ipynb)**  
   - This notebook applies unsupervised dimensionality reduction techniques (e.g., PCA, t-SNE) on embeddings to visualize them in lower dimensions.

3. **[Iris Anomaly Detection](Unsupervised/AnomalyDetection/IrisAnomalyDetection.ipynb)**  
   - This notebook applies anomaly detection algorithms such as Isolation Forest or OneClassSVM on the Iris dataset.

## Feedforward Neural Networks (FNN)

1. **[News Topic Classification](FNN/Classification/News_Classification.ipynb)**  
   - A feedforward neural network is trained to classify news articles into different topics based on a given dataset.

1. **[Content-Based Movie Recommender](FNN/Recommender/MovieRecommender-ContentBasedFiltering.ipynb)**  
   - Content-based recommender systems using methods like semantic vector search, embedding models and pairwise cosine similarity.

1. **[Collaborative-Filtering Movie Recommender](FNN/Recommender/MovieRecommender-CollaborativeFiltering.ipynb)**  
   - A neural network-based collaborative filtering model used to recommend movies based on user ratings.

---

## Convolutional Neural Networks (CNN)

1. **[CIFAR-10 Classification](CNN/ImageClassification/CIFAR-10.ipynb)**  
   - A convolutional neural network is used to classify images from the CIFAR-10 dataset into different categories.

1. **[Clothing Products Classification](CNN/ImageClassification/Fashion_MNIST.ipynb)**  
   - A convolutional neural network is used to classify clothing products images into different categories.

1. **[Car Object Detection](CNN/ObjectDetection/CarsDetection.ipynb)**  
   - This notebook uses CNNs for detecting and classifying cars in images, employing object detection techniques.

---

## Recurrent Neural Networks (RNN)

1. **[IMDB Sentiment Analysis](RNN/SentimentAnalysis/IMDB_Reviews.ipynb)**  
   - An RNN is used to analyze the sentiment of movie reviews from the IMDB dataset, classifying them as positive or negative.

1. **[Lyrics Generator](RNN/TextGeneration/LyricsGenerator.ipynb)**
   - This notebook demonstrates how to train an RNN-based lyrics generator model using dynamically fetched song lyrics.
   
---

## Reinforcement Learning (RL)

1. **[Toy Text Problems](RL/ToyText_Problems.ipynb)**  
   - DQN agents are trained to solve different toy text problems such as Taxi-v3 and Frozen Lake.

---

## Generative AI (GenAI)

1. **[Order Bot](GenAI/PromptEng/OrderBot.ipynb)**  
   - This notebook explores the process of prompt engineering for a generative AI-based chatbot that handles orders for a pizza restaurant.

1. **[PDF RAG](GenAI/RAG/PDF_RAG.ipynb)**  
   - This notebook showcases how to build a Retrieval-Augmented Generation model with LangChain to enhance text generation with information retrieved from PDF documents.

---

## Setup Instructions

To run the notebooks in this repository, ensure that you have the following dependencies installed:

```bash
pip install -r requirements.txt
```


