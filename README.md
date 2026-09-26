# ECG Arrhythmia Detection Using Weighted Federated Learning

An AI-based ECG arrhythmia detection system using **Weighted Federated Learning**, **CNN-BiGRU**, and **Flask** for privacy-preserving cardiac diagnosis.

## Overview

This project detects different types of ECG arrhythmias using Deep Learning and Federated Learning. Multiple clients can collaboratively train a global model without sharing their raw ECG data.

## Objectives

* Detect ECG arrhythmias using Deep Learning.
* Implement **Weighted Federated Learning** for privacy-preserving training.
* Keep raw ECG data local to individual clients.
* Provide ECG prediction through a Flask web application.

## Technologies Used

* Python
* PyTorch
* CNN
* BiGRU
* Federated Learning
* Flask
* HTML/CSS
* Jupyter Notebook

## Model

The project uses a **CNN + BiGRU** architecture for ECG classification.

```text
ECG Input → Preprocessing → CNN → BiGRU → Classification → Prediction
```

### Arrhythmia Classes

* Normal
* Supraventricular
* Ventricular
* Fusion
* Unknown / Other

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** for training and evaluation.

The original dataset is not included in this repository because of its large size.

## Results

The model achieved approximately **95.66% accuracy** in the reported experimental setup.

## Project Structure

```text
ECG-Arrhythmia-Federated-Learning/
│
├── app.py
├── Data-Preparation.ipynb
├── Notebook.ipynb
├── requirement-for-FL.txt
├── models/
├── static/
├── templates/
├── ecg_images/
└── .gitignore
```

## Installation & Run

### Clone the repository

```bash
git clone https://github.com/ayeshamariyam684-web/ECG-Arrhythmia-Federated-Learning.git
cd ECG-Arrhythmia-Federated-Learning
```

### Install dependencies

```bash
pip install -r requirement-for-FL.txt
```

### Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Future Enhancements

* Explainable AI using SHAP and Grad-CAM
* Transformer-based ECG models
* Real-time ECG monitoring
* Mobile application integration
* Cloud deployment
