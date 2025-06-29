# 🧠 Liver Cirrhosis Prediction Web Application

This project is a Flask-based web application that predicts the **stage of liver cirrhosis** using clinical lab data. It leverages a **Random Forest machine learning model** trained on patient records to deliver real-time, data-driven insights.

---

## 📌 Features

- 🧪 Predict cirrhosis stage (1 to 4) using clinical test results
- 🔍 Uses a trained Random Forest classifier
- 📉 Displays model accuracy and feature impact
- 🛠️ Web interface for user-friendly input
- ⚠️ Includes medical disclaimer

---

## 🧬 About the Model

This application uses a **Random Forest Classifier** trained on a modified version of the **Indian Liver Patient Dataset (ILPD)**. The model takes into account various liver function test parameters and patient age to determine the likely stage of liver cirrhosis.

### 🎯 Prediction Classes:
- **Stage 1** – Mild
- **Stage 2** – Moderate
- **Stage 3** – Severe
- **Stage 4** – End-Stage

---

## 🧑‍⚕️ Input Features

The following features are required for prediction:

- Age (years)
- Total Bilirubin (mg/dL)
- Direct Bilirubin (mg/dL)
- Alkaline Phosphotase (U/L)
- Alamine Aminotransferase (SGPT, U/L)
- Aspartate Aminotransferase (SGOT, U/L)
- Total Proteins (g/dL)
- Albumin (g/dL)
- Albumin and Globulin Ratio

---

## ⚙️ Technical Details

### 🧾 Dataset
- **Name**: Indian Liver Patient Dataset (ILPD)
- **Records**: 583 patient samples
- **Source**: Kaggle / UCI Machine Learning Repository

### ⚙️ Model Architecture
- Algorithm: Random Forest Classifier
- Trees: 100
- Max Depth: 10
- Splitting Criterion: Gini Impurity

### 🧪 Preprocessing
- Missing columns (e.g., `Direct_Bilirubin`, `SGPT`) are synthesized from existing ones.
- All numeric inputs are normalized using `StandardScaler`.

### 📊 Feature Importance
Top contributing features:
1. Albumin and Globulin Ratio
2. Total Proteins
3. Albumin
4. Age
5. Alkaline Phosphotase

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Musharaf-29/liver_cirrhosis_app.git
cd liver-cirrhosis-prediction


