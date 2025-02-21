# ❤️ Heart Disease Prediction

## 📌 Project Overview
This project focuses on predicting the presence of heart disease based on various medical attributes using machine learning techniques. The dataset contains multiple health indicators that influence heart disease risk, which are used to train a classification model.

## 📊 Dataset
The dataset contains multiple features, including:
- **👤 Age**
- **⚧️ Sex**
- **⚕️ Chest Pain Type (CP)**
- **🩸 Resting Blood Pressure (BP)**
- **🧪 Cholesterol Level**
- **💓 Fasting Blood Sugar (FBS)**
- **📈 Resting Electrocardiographic Results (ECG)**
- **🏃 Maximum Heart Rate Achieved (Thalach)**
- **🛑 Exercise Induced Angina (Exang)**
- **🩸 ST Depression Induced by Exercise (Oldpeak)**
- **📉 Slope of the Peak Exercise ST Segment**
- **🔗 Number of Major Vessels Colored by Fluoroscopy (Ca)**
- **🫀 Thalassemia (Thal)**
- **❤️ Presence of Heart Disease (Target Variable)**

## 🛠 Dependencies
The following Python libraries are required to run the notebook:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 🚀 Installation
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🔄 Data Processing
1️⃣ Load the dataset using Pandas.  
2️⃣ Perform data cleaning by handling missing values.  
3️⃣ Normalize or scale features if needed.  
4️⃣ Split the dataset into training and testing sets.  

## 💡 Model Training
- The **Random Forest Classifier** model is used for classification.
- The dataset is split into **training (80%) and testing (20%)** sets.
- The model is trained on the training dataset.

## 📈 Model Evaluation
- The performance of the trained model is evaluated using:
  - **✔️ Accuracy Score**
  - **📊 Classification Report**
  - **🗂 Confusion Matrix**
- The evaluation metrics are printed as output.

## ▶️ How to Run
1️⃣ Open the `Heart_Disease_Prediction.ipynb` file in **Jupyter Notebook** or **Google Colab**.  
2️⃣ Execute the cells sequentially to process the data and train the model.  
3️⃣ Observe the performance metrics of the model.  

## 🔥 Future Enhancements
- Implement **additional machine learning models** (e.g., **Logistic Regression, XGBoost**) for comparison.  
- Improve **feature engineering** for better predictions.  
- Tune **hyperparameters** for better accuracy.  
- Deploy the model as a **web application** for user interaction.  

## 🤝 Contribution
Feel free to contribute by **opening an issue** or **submitting a pull request**.  

## 📜 License
This project is licensed under the **MIT License**.

