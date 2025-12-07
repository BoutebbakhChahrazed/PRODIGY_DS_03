# Prodigy InfoTech Task 03: Bank Marketing Prediction

This repository contains the solution for **Task 03** of the Prodigy InfoTech Data Science Internship. The project focuses on building a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

## 📌 Project Overview
The objective of this project is to analyze the **Bank Marketing dataset** and build a predictive model to identify potential customers for term deposits. Using a **Decision Tree Classifier**, the model classifies clients into two categories:
- **Yes:** The client will subscribe to the term deposit.
- **No:** The client will not subscribe.

## 📂 Dataset
The dataset used is the **Bank Marketing Data Set** from the UCI Machine Learning Repository.
- **Source:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Description:** The data relates to direct marketing campaigns (phone calls) of a Portuguese banking institution.
- **Input Variables:**
  - **Client Data:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
  - **Last Contact:** `contact`, `month`, `day_of_week`, `duration`
  - **Other Attributes:** `campaign`, `pdays`, `previous`, `poutcome`
  - **Socio-economic:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **Target Variable:** `y` (has the client subscribed a term deposit?)

## 🛠️ Technologies Used
- **Python 3.11+**
- **Pandas:** Data manipulation and loading
- **NumPy:** Numerical operations
- **Scikit-learn:** Model building (Decision Tree), preprocessing, and evaluation metrics
- **Matplotlib & Seaborn:** Data visualization and tree plotting

## 📊 Methodology
1. **Data Loading:** Implemented robust loading to handle multiple potential file paths (Kaggle/Local).
2. **Exploratory Data Analysis (EDA):**
   - analyzed the distribution of the target variable (imbalanced classes).
   - Checked for missing values and data integrity.
   - Summarized numerical and categorical features.
3. **Preprocessing:**
   - Encoded categorical variables using `LabelEncoder`.
   - Split the dataset into training and testing sets.
4. **Model Implementation:**
   - Trained a **Decision Tree Classifier**.
   - Compared splitting criteria: **Gini Impurity** vs. **Entropy**.
   - Visualized the decision tree structure.
5. **Evaluation:**
   - Assessed performance using Accuracy, Confusion Matrix, Classification Report, F1-Score, and ROC-AUC.

## 📈 Key Results
The Decision Tree model performed well, effectively distinguishing between customers who would subscribe and those who would not.

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **92.05%** |
| **ROC AUC Score** | **0.94** |
| **F1 Score** | **0.61** |

*Note: The high ROC AUC score (0.94) indicates excellent class separability, despite the imbalanced nature of the dataset.*

## 🚀 How to Run
pip install pandas numpy scikit-learn matplotlib seaborn
3. Open the Jupyter Notebook `prodigy-ds-03.ipynb`.
4. Run the cells sequentially to load data, train the model, and view results.

## 📜 License
This project is created for educational purposes as part of the Prodigy InfoTech Data Science Internship.


---
**Author:** [Boutebbakh Chahrazed]


1. Clone the repository.
2. Install the required dependencies:
 
