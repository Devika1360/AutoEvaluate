# Auto Evaluate – Used Car Price Prediction

## Deployment Link: https://autoevaluate-9wsj2gndsri6tm2azstc9e.streamlit.app/
## Overview

**Auto Evaluate** is a machine learning project that predicts used-car prices using four regression models: **Linear Regression, Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbours (KNN) Regressor**. It uses `Car_Raw_Data.csv` and includes data cleaning, exploratory data analysis (EDA), model comparison, basic hyperparameter tuning, evaluation, and a Streamlit frontend that uses a saved model pipeline to estimate a car's price.

---

## Objective

The goal of this project is to:

* Predict the **resale price of used cars**
* Understand how car features relate to vehicle prices
* Practice data cleaning, preprocessing, and exploratory data analysis
* Train and compare four regression models
* Use cross-validation and basic hyperparameter tuning to try to improve performance
* Build an end-to-end ML workflow from raw data to a simple user interface

---

## Tech Stack

* **Python** – Main programming language
* **Pandas & NumPy** – Data loading, cleaning, and numerical operations
* **Matplotlib & Seaborn** – Data visualization and EDA
* **Scikit-learn** – Preprocessing, model training, cross-validation, tuning, and evaluation
* **Joblib** – Saving and loading the trained model pipeline
* **Streamlit** – Frontend for entering car details and viewing predicted prices
* **Jupyter Notebook / Google Colab** – Model development and experimentation

---

## Dataset

* **File:** `Car_Raw_Data.csv`
* **Source:** Kaggle
* **Dataset reference:** [Used Car Price Prediction Dataset by Taeef Najib](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)
* **Raw dataset size:** The project file used during development contains 4,345 records and 9 columns before cleaning. The number of rows used for training can be lower after removing duplicates and invalid or incomplete target values.
* **Target variable:** `Price`

The dataset contains vehicle information such as:

* Brand
* Model
* Year
* Body style
* Mileage
* Engine volume
* Engine type
* Registration status
* Price (target variable)

Use the row counts shown by the final notebook run in the report rather than assuming the raw and cleaned datasets have the same number of records.

---

## Project Workflow

### 1. Data Loading and Inspection

* Load the CSV file using `pd.read_csv()`
* Inspect the dataset using `head()`, `tail()`, `shape`, `info()`, and `describe()`
* Check column data types, missing values, and duplicate rows

### 2. Data Cleaning

* Convert relevant columns to suitable data types
* Check and handle missing values
* Remove duplicate records where appropriate
* Check for invalid or non-positive target prices
* Investigate unusual values rather than automatically deleting every outlier

### 3. Exploratory Data Analysis (EDA)

* Examine the distribution of car prices
* Explore numerical and categorical features
* Use histograms, count plots, and correlation plots where appropriate
* Study possible relationships between vehicle features and price

### 4. Feature Preparation and Preprocessing

* Separate input features (`X`) from the target variable (`y`)
* Split the data into training and test sets
* Fill missing numerical values using the median
* Fill missing categorical values using the most frequent category
* Apply one-hot encoding to categorical features
* Scale numerical features where needed, especially for KNN
* Use Scikit-learn pipelines to keep preprocessing consistent between training and prediction

### 5. Model Building

The project compares four regression models:

* **Linear Regression** – baseline regression model
* **Decision Tree Regressor** – learns rule-based, non-linear patterns
* **Random Forest Regressor** – combines multiple decision trees
* **KNN Regressor** – predicts from similar data points and benefits from feature scaling

### 6. Cross-Validation and Hyperparameter Tuning

* Compare baseline models using cross-validation
* Try a small range of hyperparameter values for selected models
* Tune Decision Tree, Random Forest, and KNN using training data
* Select the final model based on cross-validation results, not by choosing whichever happens to score highest on the test set

### 7. Model Evaluation

Models are evaluated using:

* **R² Score** – measures how much variation in car prices is explained by the model; higher is generally better
* **MAE (Mean Absolute Error)** – average absolute difference between predicted and actual prices; lower is better
* **RMSE (Root Mean Squared Error)** – penalizes larger prediction errors more heavily; lower is better

MAE and RMSE are reported in the same price units as the target variable. The final values should be taken from the final notebook execution.

### 8. Model Saving

* Save the final fitted preprocessing-and-model pipeline using `joblib`
* Save model comparison results and prediction tables as CSV files
* Reuse the saved pipeline in the Streamlit app so the same preprocessing is applied to user inputs

### 9. Frontend Development

* Built the frontend using **Streamlit**
* Used a black-and-purple visual theme with a car hero image and subtle floating animation
* Added a simple form for vehicle details
* Displayed a predicted price and a summary of the selected vehicle
* Added a model R² comparison section

---

## Model Performance

The notebook generates model comparison results after training and tuning. The results may vary depending on the final data cleaning, train-test split, and tuning run.

The following metrics are reported:

* **R² Score**
* **MAE**
* **RMSE**

Use the values produced by your final notebook run here. Do not use results from an older run if you have retrained the models.

The Streamlit app's R² cards read from `artifacts/Model_Comparison_Results.csv`; replace that file with the latest model comparison output before final submission.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd AutoEvaluate
```

Replace `<your-github-repository-url>` with the URL of your actual GitHub repository.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

### 4. Open in Browser

Streamlit normally displays a local address such as:

```text
http://localhost:8501
```

Open the address shown in your terminal.

---

## Example Prediction Flow

1. User selects the vehicle brand and model
2. User enters the year, mileage, engine volume, body style, engine type, and registration status
3. The app creates a one-row DataFrame containing the selected features
4. The saved preprocessing-and-model pipeline processes the inputs
5. The model predicts the estimated price
6. The frontend displays the estimated price and a summary of the selected vehicle

The exact transformations are handled by the fitted pipeline saved from the notebook. The frontend does not manually repeat encoding or scaling steps.

---

## Challenges Faced

* Handling missing and duplicate records
* Understanding skewed price distributions and unusual values
* Preparing categorical features for machine learning
* Making preprocessing consistent between model training and frontend prediction
* Comparing models fairly using cross-validation and a held-out test set
* Trying basic hyperparameter tuning without making the code unnecessarily complicated

---

## Future Improvements

* Try additional models if permitted by the syllabus
* Expand hyperparameter tuning carefully
* Analyze errors across different brands, vehicle ages, and price ranges
* Add a prediction interval or estimated price range
* Improve handling of rare vehicle categories
* Deploy the Streamlit app online
* Add automated tests and a reproducible environment file

---

## Project Team

* **Devika Gadge** – 47
* **Laghima Kejriwal** – 34
* **Pradyun Khot** – 59
* **Section:** A4

---

## Acknowledgements

* Kaggle for providing the used-car dataset
* Scikit-learn documentation for model training and evaluation tools
* The Python open-source community

---

## Dataset Reference

[Used Car Price Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

Please follow the dataset provider's licence and attribution requirements when redistributing the dataset.

---

## Note

This project was developed for a Machine Learning Lab mini project and is intended for educational use. Predictions are estimates based on patterns learned from the available dataset; they are not guaranteed market valuations.

## Auto Evaluate – ML Pipeline Flow

### 1. Data Collection

* Use `Car_Raw_Data.csv` as the raw used-car dataset
* Features include brand, model, year, body style, mileage, engine volume, engine type, and registration status

### 2. Data Loading

* Load data using Pandas `read_csv()`
* Inspect using `head()`, `info()`, `describe()`, and `shape`

### 3. Data Cleaning

* Check missing values and duplicate rows
* Convert relevant columns to appropriate data types
* Remove invalid target prices and handle incomplete rows according to the notebook's cleaning rules

### 4. Exploratory Data Analysis

* Plot the price distribution and examine numerical and categorical features
* Investigate outliers and relationships between features and price
* Use the plots to guide cleaning decisions rather than automatically deleting all extreme values

### 5. Feature Preparation

* Separate `Price` from the input features
* Fill missing numerical and categorical values
* One-hot encode categorical features
* Scale numerical features for models that need it, especially KNN

### 6. Train-Test Split

* Split the dataset into training and test sets
* Use the training data for model fitting, cross-validation, and hyperparameter tuning
* Keep the test set separate for final evaluation

### 7. Model Building

* Train Linear Regression
* Train Decision Tree Regressor
* Train Random Forest Regressor
* Train KNN Regressor

### 8. Cross-Validation and Tuning

* Compare baseline models using cross-validation
* Try basic hyperparameter settings for Decision Tree, Random Forest, and KNN
* Choose the final model based on cross-validation performance

### 9. Prediction and Evaluation

* Generate predictions on the held-out test data
* Calculate R², MAE, and RMSE
* Plot actual versus predicted prices and inspect residuals

### 10. Model Saving

* Save the final fitted pipeline as `best_used_car_price_pipeline.joblib`
* Save comparison results and prediction outputs as CSV files

### 11. Frontend Integration

* Load the saved model pipeline in `app.py`
* Collect vehicle details through Streamlit input controls
* Predict the price using the trained pipeline
* Display the predicted price and model comparison information

### 12. End of Pipeline

* Review the final results
* Update this README with metrics from the final run
* Test the app before submission
