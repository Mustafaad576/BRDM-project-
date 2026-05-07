Credit Default Prediction with XGBoost and SMOTE
This project implements a machine learning pipeline to predict credit default risk using the Home Credit Default Risk dataset. It addresses class imbalance using SMOTE, utilizes XGBoost for classification, and provides model interpretability via SHAP.

🚀 Getting Started
Prerequisites
The code requires the application_train.csv file. You can download it from Kaggle's Home Credit Default Risk competition.

Environment Setup
You will need the following libraries:

pandas, numpy (Data handling)

matplotlib, seaborn (Visualization)

scikit-learn (Preprocessing and Metrics)

xgboost (Modeling)

imbalanced-learn (SMOTE)

shap (Model Explainability)

🛠️ How to Run
Option 1: Google Colab (Recommended)
Open Google Colab.

Upload your notebook file (.ipynb) or copy-paste the code into a new cell.

Upload the Dataset: Click the folder icon on the left sidebar and upload application_train.csv.

Run the first cell to install dependencies:

Bash
!pip install shap xgboost imbalanced-learn
Execute all cells. The script will automatically generate and save 5 high-resolution (300 DPI) analysis graphs to your Colab file directory.

Option 2: Local Jupyter Notebook
Ensure you have Python installed (3.8+ recommended).

Clone this repository or download the script.

Install the required packages via terminal/command prompt:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap

4.  Place `application_train.csv` in the same directory as your notebook.
5.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
6.  Open the file and run all cells.

---

## 📊 Pipeline Overview

*   **Feature Selection:** Focuses on key indicators like `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, and `EXT_SOURCE` scores.
*   **Preprocessing:** Handles missing data using median imputer.
*   **Resampling:** Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the `TARGET` classes, preventing model bias toward non-defaulters.
*   **Modeling:** Employs an **XGBoost Classifier** tuned for binary classification.
*   **Interpretability:** Uses **SHAP values** to visualize which features most significantly impact the model's decision-making process.

---

## 📈 Output Graphs
The script generates five visual reports:
1.  **Class Distribution:** Comparison of labels before and after SMOTE.
2.  **Confusion Matrix:** Visual breakdown of True Positives, True Negatives, False Positives, and False Negatives.
3.  **ROC Curve:** Illustrates the diagnostic ability of the classifier (AUC-ROC score).
4.  **Precision-Recall Curve:** Evaluates the trade-off between precision and recall for the minority class.
5.  **SHAP Summary Plot:** High-level overview of feature importance and how each feature influences the prediction.

---

## 📝 License
This project is for educational purposes. The dataset is subject to the terms and conditions (may also change from the current vers)
