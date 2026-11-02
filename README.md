# 🌿 Plant Disease Detection Using Deep Learning

An AI-powered plant disease detection system that uses deep learning to analyze plant leaf images and help identify potential plant diseases.

The project combines a trained deep learning model with a Streamlit web application, allowing users to upload leaf images and obtain model predictions through a simple interface.

## 📖 Project Overview

Plant diseases can negatively affect crop quality, agricultural productivity, and farmers' income. Identifying diseases early can help farmers investigate problems and take appropriate action.

Traditional plant disease identification often requires visual inspection and agricultural expertise. This project explores how deep learning and image classification can assist with the initial identification of plant diseases.

The system uses a trained image classification model to analyze uploaded plant leaf images and generate predictions through a web-based interface built with Streamlit.

The project covers key stages of a machine learning workflow, including:

- Dataset preparation and image preprocessing.
- Deep learning model development and training.
- Model saving and reuse for inference.
- Image classification through a web application.
- Deployment of the application using Streamlit Community Cloud.

This project demonstrates the practical application of deep learning in agricultural technology.

## 🎯 Problem Statement

Manual identification of plant diseases can be time-consuming and may require specialized knowledge.

The objective of this project is to develop an image-based system that can analyze plant leaf photographs and generate predictions using a trained deep learning model.

The application aims to make model inference accessible through a simple interface without requiring users to execute Python scripts manually.

## 🚀 Project Objectives

1. Develop a deep learning-based plant leaf classification system.
2. Prepare image data for model training.
3. Train and save a model for subsequent predictions.
4. Build an interactive web application using Streamlit.
5. Integrate the trained model into the application.
6. Deploy the application so that users can access it through a web browser.
7. Explore the practical use of computer vision in agriculture.

## ✨ Key Features

- **Leaf Image Upload:** Upload plant leaf images for analysis.
- **Deep Learning Inference:** Use a trained model to generate predictions.
- **Image Preprocessing:** Prepare uploaded images for model input.
- **Interactive Web Interface:** Access the prediction workflow through Streamlit.
- **Reusable Trained Model:** Load a saved model without retraining it for every prediction.
- **Web Deployment:** Access the application through a publicly available web interface.

The exact prediction labels, confidence scores, and supported image formats depend on the implementation of the current application.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and inference |
| NumPy | Numerical operations and array processing |
| Streamlit | Web application and user interface |
| Jupyter Notebook | Model development and experimentation |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Web application deployment |

## 📂 Project Structure

The repository is organized into the following directories:

```text
Plant-Disease-Detection/
│
├── App/
│   └── app.py
│
├── Data/
│   └── Dataset files
│
├── Model/
│   └── Trained model files
│
├── Notebook/
│   └── Model development and training notebooks
│
└── README.md
```

## ⚙️ How It Works

The application follows a typical deep learning image classification workflow.

```text
        Plant Leaf Image
                |
                v
          Image Upload
                |
                v
       Image Preprocessing
                |
                v
        Trained Deep
        Learning Model
                |
                v
         Model Inference
                |
                v
         Predicted Class
                |
                v
       Results Displayed
          in Streamlit
```

### Step 1: Image Input

The user uploads a photograph of a plant leaf through the web interface.

### Step 2: Image Preprocessing

The application prepares the uploaded image in the format expected by the trained model. Depending on the implementation, this may include resizing, color conversion, and pixel normalization.

### Step 3: Model Inference

The preprocessed image is passed to the trained deep learning model, which generates its prediction.

### Step 4: Prediction Output

The application displays the prediction returned by the model.

The output depends on the model architecture, the classes used during training, and the inference logic implemented in the application.

## ⚠️ Limitations

This project is an educational deep learning application and should not be treated as a definitive agricultural diagnostic tool.

Its predictions may be affected by:

- Image quality, lighting, and camera conditions.
- Background objects and image composition.
- Disease classes represented in the training dataset.
- Differences between training images and real-world photographs.
- Model generalization and classification errors.

A prediction should be independently verified before making decisions about pesticide use, crop treatment, or other agricultural interventions.
