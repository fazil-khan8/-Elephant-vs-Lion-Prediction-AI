# 🐘🦁 Elephant vs Lion Prediction AI

A deep learning image classifier that distinguishes between **Elephants** 🐘 and **Lions** 🦁 using PyTorch and ResNet18, deployed with a Streamlit web app for real-time predictions.

---

## 📸 Application Screenshots

### Streamlit Web Interface

> Upload any elephant or lion image and get instant AI predictions!

---

## 🛠️ Technologies Used

- Python
- PyTorch
- ResNet18 (Transfer Learning)
- Streamlit
- NumPy
- Pillow

---

## 📊 Dataset

The model is trained on a custom **Elephant vs Lion** image dataset.

Dataset details:

- 🐘 Elephant images
- 🦁 Lion images

The dataset is divided into training and validation sets for model evaluation.

---

## 🧠 Model Architecture

The model uses **transfer learning** with ResNet18.

Pipeline:

```
Image → ResNet18 Feature Extractor → Global Average Pooling → Dense Layer → Binary Classification
```

ResNet18 is pretrained on the ImageNet dataset and provides powerful feature extraction for image recognition tasks.

---

## 📈 Model Performance

Training results:

- ✅ Training Accuracy: ~98%
- ✅ Training Loss: ~0.24

These results show strong performance for binary image classification.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/fazil-khan8/-Elephant-vs-Lion-Prediction-AI.git
cd "-Elephant-vs-Lion-Prediction-AI"
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare your dataset

Organize your dataset folder like this:

```
dataset/
├── elephant/
│   ├── image1.jpg
│   └── ...
└── lion/
    ├── image1.jpg
    └── ...
```

### Train the model

```bash
python train_model.py
```

### Run the app

```bash
streamlit run app.py
```

---

## 🔍 Prediction Example

Upload a clear image of an elephant or lion and the AI will instantly predict which animal it is!

---

## 👨‍💻 Author

**fazil-khan8** — BCA Student | AI & Deep Learning Enthusiast

---

⭐ If you found this project helpful, please give it a star!
