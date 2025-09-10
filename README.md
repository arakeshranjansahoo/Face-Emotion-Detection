# Face Emotion Detection

Face Emotion Detection is a deep learning project that classifies human facial expressions into emotions such as **Happy, Sad, Angry, Surprise, and Neutral** using a trained CNN model.  
It includes the training workflow, model file, and an application script to test real-time predictions.

---

## 📂 Project Structure

Face-Emotion-Detection/
├── FED.zip                   # Main project archive (app, model, notebook)
├── Train vs valid.png        # Training vs Validation performance
├── output(sad).jpg           # Example output of prediction
└── README.md                 # Project documentation

## 🚀 Features

- Real-time emotion detection using deep learning.  
- Pre-trained Keras model (`emotion_detection_model.keras`).  
- Jupyter Notebook (`ML.ipynb`) for training and evaluation.  
- Application script (`app.py`) to test predictions.  
- Visuals of training history and sample output.  


## 🛠️ Tech Stack

- **Python 3**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **Flask / Streamlit** (depending on app implementation)  
- **Matplotlib & Numpy**  


## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/arakeshranjansahoo/Face-Emotion-Detection.git
   cd Face-Emotion-Detection


2. Extract the project files:

   ```bash
   unzip FED.zip -d fed_project
   cd fed_project
   

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python app.py
   ```


## 📊 Training vs Validation

The following graph shows the training and validation performance during model training:

![Training vs Validation](./Train%20vs%20valid.png)


## 📸 Sample Output

Example prediction on a test image (detected as **Sad**):

![Sample Output](./output\(sad\).jpg)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request.

---
