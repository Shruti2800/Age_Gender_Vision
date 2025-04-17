# Age_Gender_Vision
A deep learning-based real-time gender and age recognition system using the UTKFace dataset and OpenCV.

# AgeGenderVision 👤📊

A real-time deep learning system for **gender** and **age group** recognition from facial images using the UTKFace dataset. This mini project leverages CNN models trained with Keras and uses OpenCV for live webcam predictions.

---

## 🧠 Features

- Age group classification (8 categories)
- Gender classification (Male/Female)
- Real-time webcam-based prediction using OpenCV
- Training history and confusion matrix visualization
- Preprocessing and augmentation with `ImageDataGenerator`

---

## 📁 Dataset

This project uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/) which contains over 20,000 face images labeled with age, gender, and ethnicity.

Make sure to update `DATASET_PATH` in the code with your local dataset directory:
python
DATASET_PATH = r"C:\path\to\UTKFace"

Installation
Prerequisites
Python 3.8+

pip packages:

bash
Copy
Edit
pip install tensorflow opencv-python matplotlib numpy tqdm scikit-learn
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/AgeGenderVision.git
cd AgeGenderVision
Update the dataset path in project.py.

Run the main script:

bash
Copy
Edit
python project.py
First time: it will train both models (age and gender) and save them in models/.

Next time: you can choose to load existing models.

To run live prediction:

At the end of script execution, type y when prompted:

sql
Copy
Edit
Run real-time detection? (y/n): y
🧪 Model Architecture
Both age and gender classifiers use the following CNN architecture:

3 Convolutional layers

MaxPooling after each conv

Dense layer with Dropout

Output layer with softmax activation

📊 Outputs
Accuracy/Loss plots for both models

Confusion matrices

Prediction result images

Real-time video window with detected face, predicted age group and gender

📷 Sample Output

📁 Directory Structure
lua
Copy
Edit
.
├── project.py
├── models/
│   ├── age_model.h5
│   └── gender_model.h5
├── output/
│   ├── age_model_history.png
│   ├── gender_model_history.png
│   └── prediction_results.png
