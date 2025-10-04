# Stock-Price-Prediction-using-LSTM-Ensemble
This project applies Long Short-Term Memory (LSTM) networks for time-series forecasting of stock prices, using historical financial data from Yahoo Finance. Multiple deep learning models were trained to capture temporal dependencies, and their predictions were further enhanced through ensemble methods (stacking and averaging) for improved accuracy and robustness.

**Key Features:**
1. Data collection and preprocessing from Yahoo Finance API.
2. Stock price forecasting using LSTM-based models.
3. Ensemble learning (stacking, averaging) for performance improvement.
4. Comparative analysis of individual vs ensemble predictions.
5. Visualization of actual vs predicted stock trends.

# NYC Yellow Taxi Trip Data - ML Project

## Project Description

Predicts tip amounts and fare amounts for NYC Yellow Taxi trips using machine learning models (Linear Regression and Lasso Regression).

Results:
- Fare Prediction: R² = 0.864 (86% accuracy)
- Tip Prediction: R² = 0.575 (58% accuracy)

---

## Installation

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib

Python Version: 3.8 or higher

---

## Required Files

- yellow_taxi_2023.csv (your dataset - place in project folder)
- training.ipynb (training notebook)
- test.ipynb (testing notebook)

---

## Steps to Execute

### STEP 1: Run training.ipynb

1. Import Required Libraries
   - pandas, numpy, matplotlib, seaborn
   - sklearn modules: train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, LinearRegression, Lasso, GridSearchCV
   - joblib for saving models

2. Load Dataset
   - Use pd.read_csv() to load yellow_taxi_2023.csv

3. Feature Engineering
   - Convert datetime columns using pd.to_datetime()
   - Extract pickup_day_of_week using .dt.dayofweek
   - Extract pickup_hour using .dt.hour
   - Create time_slot categories (Morning/Afternoon/Evening/Night/Other)
   - Create pre_tip_total_amount by summing fare components
   - Drop original datetime columns

4. Data Cleaning
   - Fill missing values using .fillna() with median for numerical, mode for categorical
   - Remove negative values and outliers
   - Remove rows with invalid data

5. Define Features
   - Categorical: time_slot, pickup_day_of_week, vendor_id, payment_type, ratecodeid
   - Numerical: passenger_count, trip_distance, pulocationid, dolocationid, fare components, pickup_hour
   - For tips: include pre_tip_total_amount
   - For fares: exclude pre_tip_total_amount, fare_amount, total_amount

6. Split Data
   - Use train_test_split() with 70% train, 15% validation, 15% test
   - Split twice: train vs temp (70/30), then temp into validation vs test (50/50)
   - Set random_state=42

7. Create Preprocessing Pipeline
   - Use ColumnTransformer() with:
     * StandardScaler() for numerical features
     * OneHotEncoder(drop='first', handle_unknown='ignore') for categorical features

8. Train Models
   - Create Pipeline() combining preprocessor and model
   - Train Linear Regression using LinearRegression()
   - Train Lasso Regression using Lasso()

9. Hyperparameter Tuning
   - Use GridSearchCV() with 5-fold cross-validation
   - Test alpha values: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
   - Use R² as scoring metric

10. Evaluate on Validation Set
    - Use .predict() to get predictions
    - Calculate metrics: r2_score(), mean_squared_error(), mean_absolute_error()
    - Compute 95% confidence intervals using bootstrap

11. Save Models
    - Use joblib.dump() to save:
      * linear_regression_pipeline.pkl
      * lasso_regression_pipeline.pkl
      * linear_regression_fare_pipeline.pkl
      * lasso_regression_fare_pipeline.pkl

12. Save Test Data
    - Use joblib.dump() to save:
      * X_tip_test.pkl, y_tip_test.pkl
      * X_fare_test.pkl, y_fare_test.pkl

---

### STEP 2: Run test.ipynb

1. Import Libraries
   - pandas, numpy, matplotlib, seaborn, joblib
   - sklearn.metrics: r2_score, mean_squared_error, mean_absolute_error

2. Load Trained Models
   - Use joblib.load() to load all 4 model files:
     * linear_regression_pipeline.pkl
     * lasso_regression_pipeline.pkl
     * linear_regression_fare_pipeline.pkl
     * lasso_regression_fare_pipeline.pkl

3. Load Test Data
   - Use joblib.load() to load test data files:
     * X_tip_test.pkl, y_tip_test.pkl
     * X_fare_test.pkl, y_fare_test.pkl

4. Make Predictions
   - Use .predict() method on loaded models with test data
   - No preprocessing needed (already done in training)

5. Calculate Metrics
   - Use r2_score(), mean_squared_error(), mean_absolute_error()
   - Compute 95% confidence intervals using bootstrap method

6. Generate Visualizations
   - Use matplotlib.pyplot to create:
     * R² comparison plots with error bars
     * Predicted vs Actual scatter plots
     * Residual plots
     * Feature importance plots
     * Error metrics comparison

7. Compare Models
   - Create comparison tables using pd.DataFrame()
   - Identify best model based on R² and error metrics

8. Display Results
   - Print performance metrics
   - Print confidence intervals
   - Print business insights

---

## Expected Outputs

From training.ipynb:
- 4 model pickle files (.pkl)
- 4 test data pickle files (.pkl)
- 5 visualization PNG files
- Console output with validation metrics

From test.ipynb:
- 5 test visualization PNG files
- Console output with:
  * Test R² scores with 95% CI
  * RMSE, MAE, MAPE values
  * Model comparison tables
  * Best model selection

---

## Key Functions Used

Function                  | Library  | Purpose
-------------------------|----------|----------------------------------
pd.read_csv()            | pandas   | Load dataset
pd.to_datetime()         | pandas   | Convert to datetime
train_test_split()       | sklearn  | Split data
StandardScaler()         | sklearn  | Normalize numerical features
OneHotEncoder()          | sklearn  | Encode categorical features
ColumnTransformer()      | sklearn  | Apply different transformers
Pipeline()               | sklearn  | Combine preprocessing and model
LinearRegression()       | sklearn  | Train linear model
Lasso()                  | sklearn  | Train regularized model
GridSearchCV()           | sklearn  | Tune hyperparameters
.predict()               | sklearn  | Make predictions
r2_score()               | sklearn  | Calculate R²
mean_squared_error()     | sklearn  | Calculate RMSE
joblib.dump()            | joblib   | Save models
joblib.load()            | joblib   | Load models

---

## Results Summary

Metric          | Tip Prediction | Fare Prediction
----------------|----------------|------------------
Test R²         | 0.575          | 0.864
RMSE            | $2.64          | $6.91
MAE             | $1.57          | $3.57
Best Model      | Lasso          | Linear Regression

---

## Verification

After running both notebooks, you should have:
- 8 pickle files (.pkl)
- 10 visualization files (.png)
- Test R² ≈ 0.575 for tips, ≈ 0.864 for fares

---

## Troubleshooting

Problem: Can't find dataset
Fix: Ensure file is named yellow_taxi_2023.csv in project folder

Problem: Missing .pkl files in test.ipynb
Fix: Run training.ipynb first to generate models

Problem: Memory error
Fix: Use smaller sample: df = df.sample(n=100000)

---

## Contact

[Your Name/Email]
[Date]
