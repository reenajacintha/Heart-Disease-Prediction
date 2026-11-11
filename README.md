
# Heart Disease Prediction System

## Project Overview
This project uses Machine Learning algorithms to predict the possibility of heart disease based on patient medical attributes.

The application provides an interactive interface built using Streamlit, where users can enter patient details and receive a model prediction.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Dataset
- UCI Heart Disease Dataset
- Cleveland dataset
- 303 patient records
- 13 input features

## Machine Learning Algorithms
- Logistic Regression
- Random Forest Classifier

## Model Performance
- Logistic Regression Accuracy: 86.89%
- Random Forest Accuracy: 88.52%

## Project Structure
```text
Heart-Disease-Prediction/
│
├── dataset/
│   └── processed.cleveland.data
├── app.py
├── train_model.py
├── heart_disease_model.pkl
├── scaler.pkl
├── confusion_matrix.png
├── correlation_heatmap.png
├── target_distribution.png
└── README.md
```

## How to Run the Project

1. Clone the repository:
   `git clone https://github.com/reenajacintha/Heart-Disease-Prediction.git`

2. Install dependencies:
   `pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib`

3. Run the application:
   `streamlit run app.py`

## Disclaimer
This project is developed for educational purposes only. Predictions are not medical diagnoses.
