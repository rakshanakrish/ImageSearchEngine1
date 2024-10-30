# ImageSearchEngine1
 A search engine that uses cosine similarity algorithm for finding similar images
 # Image Similarity Search with DeepImageSearch (se.py in SearchEngine)

## Overview

This project implements an image similarity search using DeepImageSearch with a pre-trained ResNet model. The application allows users to upload an image and find similar images from a specified dataset. The results include precision evaluation based on ground truth similar images.

## Features

- Image upload via a Tkinter GUI.
- Image resizing for efficient processing.
- Indexing of images using a pre-trained ResNet model.
- Retrieval of similar images based on the uploaded image.
- Precision score calculation, confusion matrix, and classification report for accuracy evaluation.
- Visualization of retrieved images and the confusion matrix using Matplotlib.

## Requirements

Make sure you have the following packages installed:

- Python 3.x
- Tkinter (usually included with Python installations)
- Pillow
- DeepImageSearch
- Matplotlib
- NumPy
- scikit-learn

You can install the required packages using pip:

```bash
pip install Pillow matplotlib numpy scikit-learn


# Image Similarity Search with MobileNetV2 (se2.py)

## Overview

This project implements an image similarity search application that utilizes the MobileNetV2 model for feature extraction. The application allows users to upload a query image and find similar images from a dataset stored in an SQLite database. The results include the product titles, image URLs, and similarity scores based on cosine similarity.

## Features

- Asynchronous loading of images for efficiency.
- Feature extraction using the MobileNetV2 model pre-trained on ImageNet.
- Storage and retrieval of extracted features using Python's pickle module.
- SQLite database integration for managing image paths and product titles.
- Visualization of query images and similar images using Matplotlib.

## Requirements

Ensure you have the following packages installed:

- Python 3.x
- TensorFlow
- Pillow
- NumPy
- scikit-learn
- Matplotlib
- SQLite (Python's built-in library)

You can install the required packages using pip:

```bash
pip install tensorflow pillow numpy scikit-learn matplotlib


