## **Project Title:**  
**Predicting Podcast Listening Time**

---

## **1. Problem Statement**

The goal of this project was to predict podcast listening times for various episodes based on features such as genre, episode length, publication time, day, number of ads, host and guest popularity, and sentiment. This is a **regression problem** where the target variable is `Listening_Time_minutes`.

---

## **2. Data Overview**

- **Training Set:** 250,000+ podcast episodes with features and the target (`Listening_Time_minutes`).
- **Test Set:** Similar structure but without the target column (used for final prediction submission).
- **Key Features:**
  - Categorical: `Genre`, `Publication_Day`, `Publication_Time`, `Episode_Sentiment`
  - Numerical: `Episode_Length_minutes`, `Host_Popularity_percentage`, `Guest_Popularity_percentage`, `Number_of_Ads`
  - Text: `Podcast_Name`, `Episode_Title` (not used directly in modeling)
  - Target: `Listening_Time_minutes` (train only)

---

## **3. Data Preprocessing**

- **Missing Value Imputation:**
  - Used **FaissImputer** (KNN-based) to fill missing values in `Episode_Length_minutes` and `Guest_Popularity_percentage`.
  - For rare cases where KNN imputation failed, fallback to median imputation.
- **Outlier Detection & Handling:**
  - Detected extreme outliers in `Episode_Length_minutes` (e.g., values > 7 million).
  - Removed two rows with implausible episode lengths to restore normal data distribution.
- **Encoding:**
  - **Label Encoding** for `Episode_Sentiment`.
  - **One-Hot Encoding** for categorical features (`Genre`, `Publication_Day`, `Publication_Time`, `Number_of_Ads`), ensuring test and train columns matched even if some categories were missing in the test set.
- **Feature Alignment:**
  - Ensured that the test set had the same columns as the train set after encoding (added missing columns with zeros if necessary).
- **Feature Engineering:**
  - Created new features such as encoded sentiment and dummies for categorical variables.

---

## **4. Exploratory Data Analysis (EDA)**

- **Visualized distributions** of key features and the target.
- **Checked for correlations** and feature relationships (e.g., strong linear relationship between episode length and listening time).
- **Analyzed categorical feature distributions** and their impact on the target.

---

## **5. Model Building**

- **Train-Test Split:**  
  Used `train_test_split` to create validation sets for model tuning.
- **Models Used:**
  - **Random Forest Regressor:** Strong baseline and best individual model.
  - **XGBoost Regressor:** Tuned with early stopping and hyperparameter search.
  - **MLPRegressor:** Neural network regressor with hyperparameter tuning via `RandomizedSearchCV`.
  - **Model Ensembling:**
    - **Simple Averaging:** Averaged predictions from RF, XGB, and MLP.
    - **Stacking Regressor:** Combined all three models with a RidgeCV meta-model, achieving the best overall validation performance.
- **Hyperparameter Tuning:**  
  Used `RandomizedSearchCV` for Random Forest and MLPRegressor to optimize key parameters.

---

## **6. Model Evaluation**

- **Metrics Used:**  
  - Mean Squared Error (MSE)
  - R² Score
- **Results on Validation Set:**

  | Model                    | MSE        | R²      |
  |--------------------------|------------|---------|
  | XGBoost                  | 178.06     | 0.7579  |
  | Random Forest            | 173.09     | 0.7647  |
  | MLPRegressor             | 186.19     | 0.7469  |
  | Simple Averaging Ensemble| 173.83     | 0.7637  |
  | Stacking Regressor       | 172.46     | 0.7655  |

- **Best Model:**  
  Stacking Regressor (ensemble of RF, XGB, MLP) with RidgeCV meta-learner.

---

## **7. Prediction and Submission**

- **Preprocessed the provided test set** identically to the training set (including imputation, encoding, and feature alignment).
- **Removed outlier rows** from the test set to match the model’s expectations.
- **Predicted `Listening_Time_minutes`** using the best model (Stacking Regressor).
- **Submitted predictions to Kaggle** in the required CSV format.

---

## **8. Kaggle Results**

- **Private Score:** 34.58026
- **Public Score:** 34.49292
- **Metric:** Root Mean Squared Error (RMSE)
- **Interpretation:** On average, predictions were within ~34.5 minutes of the true listening time.

---

## **9. Key Challenges & Solutions**

- **Handling missing and extreme values:** Used advanced imputation and outlier removal.
- **Encoding mismatches:** Ensured consistent feature columns between train and test.
- **Model selection:** Compared multiple models and ensemble strategies.
- **Leaderboard gap:** Noted that top competitors achieved much lower RMSE, likely through deeper feature engineering and more aggressive ensembling.

---

## **10. Learnings & Takeaways**

- End-to-end ML workflow: data cleaning, feature engineering, model selection, ensembling, and evaluation.
- Importance of robust preprocessing and feature alignment for real-world data.
- Value of model ensembling for boosting performance.
- Practical experience with Kaggle’s workflow and submission process.
- Identified areas for future improvement: deeper feature engineering, advanced ensembling, and leveraging GPU acceleration.

---

## **11. References**

- [Kaggle Competition Page](https://www.kaggle.com/competitions/playground-series-s5e4)
