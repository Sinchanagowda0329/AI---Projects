# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using two approaches: a cost-sensitive XGBoost classifier and an Autoencoder-based anomaly detector.

---

## Project Overview

Credit card fraud is highly imbalanced — fraudulent transactions make up less than 0.2% of all transactions. This project tackles that challenge using two complementary models and explains predictions using SHAP values.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraudulent
- **Features:** 30 features (V1–V28 are PCA-transformed, plus `Time` and `Amount`)
- **Target:** `Class` (0 = legitimate, 1 = fraud)

---

## Requirements

Install the following before running:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost shap tensorflow kaggle
```

---

## Setup & Usage

### 1. Add your Kaggle credentials
In the code, replace or keep the following lines with your own Kaggle API key:

```python
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'
```

> Get your key from: Kaggle → Settings → API Tokens → Create Legacy API Key

### 2. Run the notebook
The code will:
- Automatically download and unzip the dataset from Kaggle
- Preprocess and scale the data
- Train both models
- Print results and display SHAP plots

---

## Models

### XGBoost (Supervised)
- Uses `scale_pos_weight` to handle class imbalance
- Trained on labelled data
- Evaluated using confusion matrix and classification report

### Autoencoder (Unsupervised)
- Trained **only on normal transactions**
- Flags anomalies based on high reconstruction error
- Threshold set at the 95th percentile of MSE scores

### SHAP Explainability
- Uses `TreeExplainer` to explain XGBoost predictions
- Summary plot shows which features contribute most to fraud detection

---

## Project Structure

```
├── creditcard.csv          # Downloaded automatically from Kaggle
├── notebook.ipynb          # Main notebook with all code
└── README.md               # This file
```

---

## Results

Both models print:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

XGBoost typically achieves high precision and recall on fraud class due to cost-sensitive training. The Autoencoder provides an unsupervised baseline useful when labels are unavailable.

---

## Notes

- Keep your Kaggle API key private — do not share it publicly
- The dataset is downloaded to `/content/creditcard.csv` (Google Colab path)
- If running locally, change `file_path` to your local path

---

## License

Dataset is licensed under [DbCL-1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
