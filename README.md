Got it 👍
Here is **ONE complete, clean `README.md`**, already merged and formatted so you can **copy–paste it exactly as it is** into GitHub.

---

````md
# NBA Player Performance Analysis (AI305 Course Project)

This project applies **Machine Learning techniques** to analyze NBA player data using a complete end-to-end workflow.  
It covers **regression, unsupervised learning (clustering), and classification**, with proper evaluation metrics and visualizations.

The project was developed as part of the **AI305 course**.

---

## Project Overview

The notebook demonstrates a full machine learning pipeline:
- Data loading and preprocessing
- Feature engineering
- Training multiple ML models
- Model evaluation and comparison
- Visualization of results
- Saving trained models and generated plots

---

## Machine Learning Tasks

### Regression – Player Performance Prediction
Predicts continuous player performance values using:
- Linear Regression  
- Random Forest Regressor  
- MLP Regressor (Neural Network)

**Evaluation metrics:** MAE, MSE, R²

---

### Unsupervised Learning – Player Style Clustering
Groups players based on statistical similarity using:
- K-Means  
- Agglomerative Clustering  
- PCA for dimensionality reduction and 2D visualization  

**Evaluation:** Silhouette Score and cluster visualization

---

### Classification – Game Outcome Prediction (W/L)
Predicts win/loss outcomes using:
- Logistic Regression  
- Decision Tree  
- Random Forest Classifier (with GridSearchCV)  
- MLP Classifier  

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

---

## Project Files
- `Project AI305.ipynb` — Main Jupyter Notebook
- `nba_data.csv` — Dataset used for training and evaluation
- `models/` — Saved trained models (generated after execution)
- `plots/` — Saved visualizations (generated after execution)

---

## How to Run

1. Install required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib notebook
````

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `Project AI305.ipynb` and run all cells.

---

## Group Contribution Note

This was a **group project**, and all members contributed collaboratively.


---

## Notes

* Model training time may vary depending on hardware.
* Some steps (such as GridSearchCV) may take longer to execute.

```

